#!/usr/bin/env python3
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker


class BeadPoseNode(Node):
    def __init__(self):
        super().__init__('bead_pose_node')

        self.bridge = CvBridge()

        # ===== 카메라 Intrinsic =====
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_frame = None

        # ===== 최신 Depth 이미지 =====
        self.depth_image = None
        self.depth_header = None

        # RealSense depth scale (16UC1 -> meters)
        self.depth_scale = 0.001  # 필요하면 수정

        # ===== TF Buffer / Listener =====
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ===== Sensor QoS =====
        sensor_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # ===== Subscribers =====
        self.sub_caminfo = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            # '/camera/camera/depth/camera_info',
            self.caminfo_callback,
            qos_profile=sensor_qos,
        )

        self.sub_depth = self.create_subscription(
            Image,
            # '/camera/camera/aligned_depth_to_color/image_raw',
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            qos_profile=sensor_qos,
        )

        self.sub_mask = self.create_subscription(
            Image,
            '/bead/mask',
            self.mask_callback,
            10
        )

        # ===== Publishers =====
        self.start_pose_pub = self.create_publisher(
            PoseStamped,
            '/bead/grind_start_pose_base',
            10
        )

        self.end_pose_pub = self.create_publisher(
            PoseStamped,
            '/bead/grind_end_pose_base',
            10
        )

        self.start_marker_pub = self.create_publisher(
            Marker,
            '/bead/start_marker',
            10
        )
        self.end_marker_pub = self.create_publisher(
            Marker,
            '/bead/end_marker',
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            '/bead/grind_path_base',
            10
        )

        # 비드 포인트 클라우드 (RViz 용)
        self.bead_points_pub = self.create_publisher(
            PointCloud2,
            '/bead/points_base',
            10
        )

        # ✅ 한 번 경로 계산하면 잠그는 플래그
        self.path_locked = False

        # ✅ 외부에서 경로 리셋하고 싶을 때 사용할 서비스
        self.reset_srv = self.create_service(
            Trigger,
            'reset_bead_path',
            self.reset_bead_path_cb
        )

        self.get_logger().info('✅ BeadPoseNode started.')

    # ---------------- reset 서비스 콜백 ----------------
    def reset_bead_path_cb(self, request, response):
        self.path_locked = False
        self.get_logger().info("🔁 reset_bead_path called. path_locked=False")
        response.success = True
        response.message = "Bead grind path reset. Will recompute on next valid mask."
        return response

    # ---------------- CameraInfo 콜백 ----------------
    def caminfo_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        # 그냥 카메라가 주는 frame_id 그대로 사용
        self.camera_frame = msg.header.frame_id
        self.get_logger().info(f"CameraInfo from frame: {self.camera_frame}")

    # ---------------- Depth 콜백 ----------------
    def depth_callback(self, msg: Image):
        self.get_logger().info(
            f"[depth_callback] got depth: {msg.width}x{msg.height}, "
            f"encoding={msg.encoding}, frame={msg.header.frame_id}"
        )
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        self.depth_image = depth
        self.depth_header = msg.header

    # ---------------- Mask 콜백 (핵심) ----------------
    def mask_callback(self, msg: Image):
        self.get_logger().info("mask_callback called")

        # # ✅ 이미 경로를 한 번 만들었다면 이후 마스크는 무시
        # if self.path_locked:
        #     self.get_logger().info("path_locked=True, ignore new mask.")
        #     return

        if self.path_locked:
            self.get_logger().info("path_locked=True -> cloud only mode")
            only_cloud_mode = True      # ← 추가
        else:
            only_cloud_mode = False     # ← 추가

        # 아직 depth나 camera info 없으면 skip
        if self.depth_image is None:
            self.get_logger().warn("No depth_image yet, skip.")
            return
        if self.fx is None:
            self.get_logger().warn("No camera intrinsics yet (fx is None), skip.")
            return

        # 1) 마스크, depth 해상도 맞추기
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        dh, dw = self.depth_image.shape[:2]
        mh, mw = mask.shape[:2]
        self.get_logger().info(f"depth shape={dh}x{dw}, mask shape={mh}x{mw}")

        if (mh, mw) != (dh, dw):
            mask_resized = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        depth = self.depth_image

        # 2) 비드 픽셀 좌표(u, v) 찾기
        ys, xs = np.where(mask_resized > 0)
        self.get_logger().info(f"non-zero mask pixels: {len(xs)}")

        if len(xs) == 0:
            self.get_logger().warn("Mask has no non-zero pixels, skip.")
            return

        # 3) 해당 픽셀들의 depth 값(m) 얻기
        depths = depth[ys, xs].astype(np.float32) * self.depth_scale
        valid = depths > 0.0
        self.get_logger().info(f"valid depth pixels: {np.count_nonzero(valid)}")

        if not np.any(valid):
            self.get_logger().warn("All depth values are 0, skip.")
            return

        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        depths = depths[valid]

        # 너무 많으면 subsample
        MAX_POINTS = 5000
        if len(xs) > MAX_POINTS:
            step = len(xs) // MAX_POINTS
            xs = xs[::step]
            ys = ys[::step]
            depths = depths[::step]

        # 4) (u, v, Z) -> 카메라 좌표계 (X, Y, Z)
        Z = depths
        X = (xs - self.cx) * Z / self.fx
        Y = (ys - self.cy) * Z / self.fy

        pts_cam = np.stack([X, Y, Z], axis=1)  # (N,3)

        # 5) TF를 이용해 base_link 좌표계로 변환
        try:
            self.get_logger().info(f"Looking up TF base_link <- {self.camera_frame}")
            trans = self.tf_buffer.lookup_transform(
                'base_link',          # target
                self.camera_frame,    # source
                Time()                # latest
            )
            self.get_logger().info("TF lookup OK.")
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return

        pts_base = []
        for Xc, Yc, Zc in pts_cam:
            ps = PoseStamped()
            ps.header = self.depth_header
            ps.header.frame_id = self.camera_frame
            ps.pose.position.x = float(Xc)
            ps.pose.position.y = float(Yc)
            ps.pose.position.z = float(Zc)
            ps.pose.orientation.w = 1.0

            # Pose만 넘겨서 변환
            pose_base = do_transform_pose(ps.pose, trans)

            pts_base.append([
                pose_base.position.x,
                pose_base.position.y,
                pose_base.position.z
            ])

        pts_base = np.array(pts_base)  # (N,3)

        # 포인트가 너무 적으면 중단
        if pts_base.shape[0] < 10:
            self.get_logger().warn('Not enough 3D points after TF transform.')
            return

        # ======================
        # 6) PCA 기반으로 비드 방향 및 양 끝점(start/end) 계산
        # ======================
        xy = pts_base[:, :2]  # (N,2)

        if xy.shape[0] < 10:
            self.get_logger().warn('Not enough points for path generation.')
            return

        xy_mean = xy.mean(axis=0, keepdims=True)
        xy_centered = xy - xy_mean

        cov = np.cov(xy_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        idx_dir = int(np.argmax(eigvals))
        direction = eigvecs[:, idx_dir]
        direction = direction / np.linalg.norm(direction)

        # 각 점을 PCA 직선 상의 1D 파라미터 t로 투영
        t = xy_centered @ direction
        t_min, t_max = float(t.min()), float(t.max())

        # ✅ 비드의 "양 끝" 인덱스 (start/end)
        idx_min = int(np.argmin(t))
        idx_max = int(np.argmax(t))

        start_pt = pts_base[idx_min]
        end_pt = pts_base[idx_max]

        z_mean = float(pts_base[:, 2].mean())

        # # ======================
        # # 7) 시작/끝 PoseStamped 퍼블리시
        # # ======================
        # header_base = self.get_clock().now().to_msg()

        # start_pose = PoseStamped()
        # start_pose.header.stamp = header_base
        # start_pose.header.frame_id = 'base_link'
        # start_pose.pose.position.x = float(start_pt[0])
        # start_pose.pose.position.y = float(start_pt[1])
        # start_pose.pose.position.z = z_mean
        # start_pose.pose.orientation.w = 1.0

        # end_pose = PoseStamped()
        # end_pose.header.stamp = header_base
        # end_pose.header.frame_id = 'base_link'
        # end_pose.pose.position.x = float(end_pt[0])
        # end_pose.pose.position.y = float(end_pt[1])
        # end_pose.pose.position.z = z_mean
        # end_pose.pose.orientation.w = 1.0

        # self.start_pose_pub.publish(start_pose)
        # self.end_pose_pub.publish(end_pose)

        # # ======================
        # # RViz Marker 시각화
        # # ======================

        # # Start marker
        # start_marker = Marker()
        # start_marker.header.frame_id = 'base_link'
        # start_marker.header.stamp = self.get_clock().now().to_msg()
        # start_marker.ns = 'bead_start'
        # start_marker.id = 0
        # start_marker.type = Marker.SPHERE
        # start_marker.action = Marker.ADD
        # start_marker.pose = start_pose.pose
        # start_marker.scale.x = 0.03
        # start_marker.scale.y = 0.03
        # start_marker.scale.z = 0.03
        # start_marker.color.r = 1.0
        # start_marker.color.g = 0.0
        # start_marker.color.b = 0.0
        # start_marker.color.a = 1.0
        # self.start_marker_pub.publish(start_marker)

        # # End marker
        # end_marker = Marker()
        # end_marker.header.frame_id = 'base_link'
        # end_marker.header.stamp = start_marker.header.stamp
        # end_marker.ns = 'bead_end'
        # end_marker.id = 1
        # end_marker.type = Marker.SPHERE
        # end_marker.action = Marker.ADD
        # end_marker.pose = end_pose.pose
        # end_marker.scale.x = 0.03
        # end_marker.scale.y = 0.03
        # end_marker.scale.z = 0.03
        # end_marker.color.r = 0.0
        # end_marker.color.g = 0.0
        # end_marker.color.b = 1.0
        # end_marker.color.a = 1.0
        # self.end_marker_pub.publish(end_marker)

        # ======================
        # 8) 비드 경로(Path) 계산 (PCA 직선 따라 샘플링)
        # ======================
        NUM_WAYPOINTS = 30
        ts = np.linspace(t_min, t_max, NUM_WAYPOINTS)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'base_link'

        for ti in ts:
            xy_i = xy_mean[0] + ti * direction
            pose_st = PoseStamped()
            pose_st.header = path_msg.header
            pose_st.pose.position.x = float(xy_i[0])
            pose_st.pose.position.y = float(xy_i[1])
            pose_st.pose.position.z = z_mean
            pose_st.pose.orientation.w = 1.0
            path_msg.poses.append(pose_st)

        self.path_pub.publish(path_msg)

        # ======================
        # 9) RViz 포인트클라우드 시각화
        # ======================
        header = self.depth_header
        header.frame_id = 'base_link'
        cloud_msg = pc2.create_cloud_xyz32(header, pts_base.tolist())
        self.bead_points_pub.publish(cloud_msg)

        self.get_logger().info(
            f"📌 start=({start_pt[0]:.3f}, {start_pt[1]:.3f}, {start_pt[2]:.3f}), "
            f"end=({end_pt[0]:.3f}, {end_pt[1]:.3f}, {end_pt[2]:.3f}), "
            f"path points={len(path_msg.poses)}, cloud points={len(pts_base)}"
        )

        # ✅ 첫 경로 계산 후 잠금
        self.path_locked = True
        self.get_logger().info("✅ Bead grind path locked. Further masks will be ignored.")


def main(args=None):
    rclpy.init(args=args)
    node = BeadPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
