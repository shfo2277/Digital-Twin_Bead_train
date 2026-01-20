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
# CONFIG
# ==============================
IMAGE_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

MODEL_PATH = "/workspace/BEADtrain/modelresult/1125unet.pth"

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

class BeadUNetNode(Node):
    def __init__(self):
        super().__init__("bead_unet_node")

        self.bridge = CvBridge()
        self.get_logger().info("✅ BeadUNetNode started")

        # 모델
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(DEVICE)

        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE)
        )
        self.model.eval()

        # ROS
        self.sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_rect_raw",
            self.image_callback,
            10
        )

        self.pub_mask = self.create_publisher(
            Image,
            "/bead/mask",
            10
        )

    def image_callback(self, msg: Image):
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = frame_bgr.shape

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        img = transform(frame_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(img))
            mask = (prob > 0.5).float()

        mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

def main():
    rclpy.init()
    node = BeadUNetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
