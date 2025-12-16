#/home/ho/BEADtrain/Calibration/handeye_calib_node.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time

import numpy as np
import cv2
from cv2 import aruco


class HandEyeCalibNode(Node):
    def __init__(self):
        super().__init__('handeye_calib_node')

        # -----------------------------
        # 🔧 기본 프레임 / 파라미터 설정
        # -----------------------------
        self.base_frame = 'base_link'
        self.tcp_frame  = 'tool0'
        self.camera_frame = 'camera_color_optical_frame'


        # 🔧 네가 만든 3×3 아루코 보드 설정 (4cm, gap 1cm)
        self.marker_length = 0.04              # [m] 마커 한 변 = 4cm
        self.marker_separation = 0.01          # [m] 마커 사이 간격 = 1cm
        self.rows = 3
        self.cols = 3

        self.num_samples_target = 20
        self.sample_interval_sec = 0.7

        # -----------------------------
        # 🔧 TF 버퍼 & 리스너
        # -----------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -----------------------------
        # 🔧 카메라 구독
        # -----------------------------
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.cinfo_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.cinfo_callback,
            10
        )

        # 카메라 내참/왜곡
        self.camera_matrix = None
        self.dist_coeffs = None

        # -----------------------------
        # 🔧 ArUco 딕셔너리 & GridBoard
        # -----------------------------
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        if hasattr(aruco, "DetectorParameters_create"):
            self.aruco_params = aruco.DetectorParameters_create()
        else:
            self.aruco_params = aruco.DetectorParameters()
        self.board = aruco.GridBoard(
            (self.cols, self.rows),           # (markersX, markersY)
            self.marker_length,               # markerLength
            self.marker_separation,           # markerSeparation
            self.aruco_dict                   # dictionary
        )

        # 샘플 저장 리스트
        self.R_gripper2base_list = []
        self.t_gripper2base_list = []
        self.R_target2cam_list = []
        self.t_target2cam_list = []

        self.last_sample_time = self.get_clock().now()

        self.get_logger().info('✅ Hand-Eye GridBoard calibration node started.')
        self.get_logger().info(f'   base_frame: {self.base_frame}, tcp_frame: {self.tcp_frame}')
        self.get_logger().info('   3x3 GridBoard, marker=0.04m, gap=0.01m, DICT_5X5_100 사용 중')


    # -----------------------------
    # 📷 CameraInfo 콜백
    # -----------------------------
    def cinfo_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64)
            self.camera_matrix = K
            self.dist_coeffs = D
            self.get_logger().info('📷 Camera info received & stored.')

    # -----------------------------
    # 📷 이미지 콜백
    # -----------------------------
    def image_callback(self, msg: Image):
        # CameraInfo 준비 안됐으면 skip
        if self.camera_matrix is None or self.dist_coeffs is None:
            return
    
        # ROS Image → OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
        # 1) detect markers
        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )
        if ids is None or len(ids) == 0:
            cv2.imshow("Calibration", cv_image)
            cv2.waitKey(1)
            return
    
        # 2) estimate pose
        rvec_init = np.zeros((3, 1), dtype=np.float32)
        tvec_init = np.zeros((3, 1), dtype=np.float32)
    
        retval, rvec, tvec = aruco.estimatePoseBoard(
            corners, ids, self.board,
            self.camera_matrix, self.dist_coeffs,
            rvec_init, tvec_init
        )
        if retval <= 0:
            cv2.imshow("Calibration", cv_image)
            cv2.waitKey(1)
            return
    
        # pose OK
        R_target2cam, _ = cv2.Rodrigues(rvec)
        t_target2cam = tvec.reshape(3, 1)
    
        # 3) TF lookup (base -> tcp)
        try:
            now = Time()
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, now
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            cv2.imshow("Calibration", cv_image)
            cv2.waitKey(1)
            return
    
        q = trans.transform.rotation
        t = trans.transform.translation
    
        T_base_tcp = self.quaternion_to_matrix([q.x, q.y, q.z, q.w])
        T_base_tcp[0, 3] = t.x
        T_base_tcp[1, 3] = t.y
        T_base_tcp[2, 3] = t.z
    
        R_gripper2base = T_base_tcp[0:3, 0:3]
        t_gripper2base = T_base_tcp[0:3, 3].reshape(3, 1)
    
        # -----------------------------
        # ▶▶ HERE: S 키 누르면 샘플 저장
        # -----------------------------
        cv2.putText(cv_image, "Press 's' to save sample", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
        cv2.imshow("Calibration", cv_image)
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord('s'):  # ← S 키로 저장
            if len(self.R_gripper2base_list) < self.num_samples_target:
                self.R_gripper2base_list.append(R_gripper2base)
                self.t_gripper2base_list.append(t_gripper2base)
                self.R_target2cam_list.append(R_target2cam)
                self.t_target2cam_list.append(t_target2cam)
    
                self.get_logger().info(
                    f"📸 Sample {len(self.R_gripper2base_list)}/{self.num_samples_target} saved!"
                )
    
            if len(self.R_gripper2base_list) >= self.num_samples_target:
                self.compute_handeye()



    # -----------------------------
    # 📐 Hand-Eye 계산
    # -----------------------------
    def compute_handeye(self):
        self.get_logger().info('📐 Running cv2.calibrateHandEye (GridBoard)...')

        Rg = self.R_gripper2base_list
        tg = self.t_gripper2base_list
        Rt = self.R_target2cam_list
        tt = self.t_target2cam_list

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            Rg, tg, Rt, tt, method=cv2.CALIB_HAND_EYE_TSAI
        )

        T_tcp_cam = np.eye(4)
        T_tcp_cam[0:3, 0:3] = R_cam2gripper
        T_tcp_cam[0:3, 3] = t_cam2gripper.flatten()

        self.get_logger().info('🎉 Hand-Eye calibration result (T_tcp_cam = tcp <- cam):')
        self.get_logger().info('\nT_tcp_cam =\n' + str(T_tcp_cam))

        tx, ty, tz = t_cam2gripper.flatten()
        roll, pitch, yaw = self.rotationMatrixToRPY(R_cam2gripper)

        self.get_logger().info(
            f'\n👉 Use this in static_transform_publisher (tool0 -> camera_link)\n'
            f'xyz = ({tx:.4f}, {ty:.4f}, {tz:.4f}) [m]\n'
            f'rpy = ({roll:.4f}, {pitch:.4f}, {yaw:.4f}) [rad]\n'
        )

    # -----------------------------
    # 🔧 쿼터니언 → 4x4 행렬
    # -----------------------------
    def quaternion_to_matrix(self, q):
        x, y, z, w = q
        n = x*x + y*y + z*z + w*w
        if n < 1e-8:
            return np.eye(4, dtype=np.float64)
        s = 2.0 / n
        xx, yy, zz = x*x*s, y*y*s, z*z*s
        xy, xz, yz = x*y*s, x*z*s, y*z*s
        wx, wy, wz = w*x*s, w*y*s, w*z*s

        R = np.array([
            [1.0 - (yy + zz),     xy - wz,           xz + wy],
            [xy + wz,             1.0 - (xx + zz),   yz - wx],
            [xz - wy,             yz + wx,           1.0 - (xx + yy)]
        ], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[0:3, 0:3] = R
        return T

    # -----------------------------
    # 🔧 회전행렬(3x3) → RPY
    # -----------------------------
    def rotationMatrixToRPY(self, R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
            pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
