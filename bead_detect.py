#!/usr/bin/env python3
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

# -----------------------------
# 1) 기본 설정
# -----------------------------
IMAGE_SIZE = 1024
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class BeadUNetDepthNode(Node):
    def __init__(self):
        super().__init__("bead_unet_depth_node")

        self.bridge = CvBridge()
        self.latest_depth = None  # 최신 depth 프레임 저장

        # -----------------------------
        # 2) 모델 정의 + 가중치 로드
        # -----------------------------
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        state_dict = torch.load(
            "/home/ho/BEADtrain/modelresult/1125unet.pth",
            map_location=device,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.get_logger().info("✅ U-Net 모델 로드 완료")

        # -----------------------------
        # 3) RealSense 컬러/뎁스 토픽 구독
        # -----------------------------
        # rs_launch.py align_depth.enable:=true 기준 (필요하면 토픽 이름 ros2 topic list 로 확인)
        self.color_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            10,
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10,
        )

        # -----------------------------
        # 4) 비드 중심 + 거리 퍼블리셔
        # -----------------------------
        self.center_pub = self.create_publisher(
            PointStamped,
            "/bead/center_depth",
            10,
        )

    # ---- depth 콜백: depth 프레임만 계속 갱신 ----
    def depth_callback(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.latest_depth = depth

    # ---- color 콜백: U-Net + mask + depth 사용 ----
    def image_callback(self, msg: Image):
        if self.latest_depth is None:
            # 아직 depth 안 들어왔으면 스킵
            self.get_logger().warn("아직 depth 프레임 없음, 스킵합니다.")
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        orig_h, orig_w = frame.shape[:2]

        # BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

        img = img_resized.astype(np.float32) / 255.0
        img[..., 0] = (img[..., 0] - MEAN[0]) / STD[0]
        img[..., 1] = (img[..., 1] - MEAN[1]) / STD[1]
        img[..., 2] = (img[..., 2] - MEAN[2]) / STD[2]

        img_chw = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)

        # -------------------------
        # U-Net 추론
        # -------------------------
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits)[0, 0]
            prob_np = probs.cpu().numpy()

        mask_resized = cv2.resize(prob_np, (orig_w, orig_h))
        mask_bin = (mask_resized > 0.5).astype(np.uint8)

        # -------------------------
        # depth 기반 거리 계산
        # -------------------------
        depth = self.latest_depth
        if depth.shape[:2] != (orig_h, orig_w):
            depth = cv2.resize(
                depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

        depth = depth.astype(np.float32)
        # RealSense 기본 16UC1 이면 mm 단위 → m 로 변환
        if depth.dtype != np.float32:
            depth_m = depth / 1000.0
        else:
            depth_m = depth

        mask_indices = mask_bin > 0
        valid_depths = depth_m[mask_indices]
        valid_depths = valid_depths[valid_depths > 0.0]

        center_depth = None
        cx = cy = None

        if valid_depths.size > 0:
            # 가장 큰 contour 기준으로 중심 계산
            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) > 0:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                cx = x + w // 2
                cy = y + h // 2

                center_depth = depth_m[cy, cx]
                median_dist = float(np.median(valid_depths))
                min_dist = float(np.min(valid_depths))

                self.get_logger().info(
                    f"Bead depth: center={center_depth:.3f} m, "
                    f"median={median_dist:.3f} m, min={min_dist:.3f} m"
                )

                # 텍스트로 화면에 표시
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{center_depth:.3f} m",
                    (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # ---- ROS 토픽으로 퍼블리시 ----
                msg_center = PointStamped()
                msg_center.header.stamp = self.get_clock().now().to_msg()
                # 일단 camera frame 기준 (px, px, m). 나중에 3D로 바꾸면 X,Y,Z[m] 넣으면 됨.
                msg_center.header.frame_id = msg.header.frame_id
                msg_center.point.x = float(cx)  # pixel x
                msg_center.point.y = float(cy)  # pixel y
                msg_center.point.z = float(center_depth)  # m

                self.center_pub.publish(msg_center)

        # -------------------------
        # 시각화 (원하면 끄거나 유지)
        # -------------------------
        color_mask = np.zeros_like(frame)
        color_mask[:, :, 2] = mask_bin * 255

        overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
        cv2.imshow("Bead Segmentation - Overlay", overlay)
        cv2.imshow("Bead Mask", mask_bin * 255)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.get_logger().info("'q' 입력. 노드 종료 요청.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = BeadUNetDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
