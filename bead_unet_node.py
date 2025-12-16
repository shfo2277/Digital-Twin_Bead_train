#계속 받아오는 거 
#!/usr/bin/env python3
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ==============================
# 0) 모델 / 전처리 설정
# ==============================
IMAGE_SIZE = 1024  # 학습 때랑 동일
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# 🔧 모델 가중치 경로 (컨테이너 기준)
MODEL_PATH = "/workspace/BEADtrain/modelresult/1125unet.pth"
#MODEL_PATH = "/workspace/BEADtrain/modelresult/1127unet.pth"

class BeadUNetNode(Node):
    def __init__(self):
        super().__init__("bead_unet_node")

        self.bridge = CvBridge()
        self.get_logger().info("✅ BeadUNetNode started, waiting for camera images...")

        # -----------------------------
        # 1) smp.Unet(resnet34) 모델 생성 + 가중치 로드
        #    👉 bead_train.py와 완전히 동일하게
        # -----------------------------
        self.get_logger().info("✅ Loading Unet(resnet34) model...")

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",   # 학습 코드와 동일
            in_channels=3,
            classes=1,                    # binary segmentation
        ).to(DEVICE)

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.get_logger().info(f"✅ Model loaded from {MODEL_PATH}")
        self.get_logger().info(f"   Device: {DEVICE}")

        # -----------------------------
        # 2) 구독 & 퍼블리셔
        # -----------------------------
        # rqt_image_view에서 실제로 보이는 토픽 이름에 맞추기!
        CAMERA_TOPIC = "/camera/camera/color/image_raw"
        self.get_logger().info(f"📡 Subscribing to: {CAMERA_TOPIC}")

        self.sub_image = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10,   # QoS depth
        )

        self.pub_overlay = self.create_publisher(
            Image,
            "/bead/overlay",
            10
        )

        self.pub_mask = self.create_publisher(
            Image,
            "/bead/mask",
            10
        )

    def image_callback(self, msg: Image):
        # --- 디버그 로그 1: 들어온 이미지 정보 확인 ---
        self.get_logger().info(
            f"📩 Got image: {msg.width}x{msg.height}, encoding={msg.encoding}, frame_id={msg.header.frame_id}"
        )

        # ROS Image -> OpenCV BGR
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w, _ = frame_bgr.shape

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # 3) 전처리 (학습과 동일하게)
        # -----------------------------
        img_tensor = transform(frame_rgb)              # [3, H, W]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

        with torch.no_grad():
            logits = self.model(img_tensor)            # [1, 1, 1024, 1024]
            probs  = torch.sigmoid(logits)
            mask   = (probs > 0.5).float()             # binary mask

        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)  # 0/1
        mask_np = mask_np * 255                                      # 0/255

        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- 디버그 로그 2: 마스크 통계 보기 ---
        # self.get_logger().info(
        #     f"🧪 mask_resized: shape={mask_resized.shape}, "
        #     f"min={int(mask_resized.min())}, max={int(mask_resized.max())}, "
        #     f"nonzero={int((mask_resized>0).sum())}"
        # )
        
        # ✅ /bead/mask 퍼블리시 (mono8, 0/255)
        mask_msg = self.bridge.cv2_to_imgmsg(mask_resized, encoding="mono8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)
        self.get_logger().info("✅ Published /bead/mask")

        # -----------------------------
        # 4) Overlay 색칠하기
        # -----------------------------
        overlay = frame_bgr.copy()
        color = np.array([255, 0, 0], dtype=np.uint8)  # BGR 파란색
        alpha = 0.5

        mask_bool = mask_resized > 0
        overlay[mask_bool] = (
            alpha * color + (1 - alpha) * overlay[mask_bool]
        ).astype(np.uint8)

        # # --- 디버그 로그 3: overlay 통계 보기 ---
        # self.get_logger().info(
        #     f"🎨 overlay: shape={overlay.shape}, "
        #     f"min={int(overlay.min())}, max={int(overlay.max())}"
        # )

        # -----------------------------
        # 5) /bead/overlay 퍼블리시
        # -----------------------------
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        self.get_logger().info("✅ Published /bead/overlay")


def main(args=None):
    rclpy.init(args=args)
    node = BeadUNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
