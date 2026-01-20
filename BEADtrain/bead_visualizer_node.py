#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge


class BeadVisualizerNode(Node):
    def __init__(self):
        super().__init__("bead_visualizer_node")

        self.bridge = CvBridge()
        self.contour_saved = False
        self.saved_contour = None

        self.sub_mask = self.create_subscription(
            Image,
            "/bead/mask",
            self.mask_callback,
            10
        )

        self.pub_marker = self.create_publisher(
            Marker,
            "/bead/contour_marker",
            1
        )

        self.get_logger().info("🎨 BeadVisualizerNode started")

    def mask_callback(self, msg: Image):
        if self.contour_saved:
            return  # 🔒 한 번만 추출

        mask = self.bridge.imgmsg_to_cv2(msg, encoding="mono8")

        # contour 추출
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            self.get_logger().warn("⚠ No contour found")
            return

        largest = max(contours, key=cv2.contourArea)
        self.saved_contour = largest
        self.publish_contour_marker(largest)

        self.contour_saved = True
        self.get_logger().info("✅ Bead contour saved & visualized")

    def publish_contour_marker(self, contour):
        marker = Marker()
        marker.header.frame_id = "camera_color_optical_frame"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "bead_contour"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.002  # 선 두께
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        for p in contour:
            x, y = p[0]
            marker.points.append(
                self.make_point(x, y)
            )

        # 닫힌 선
        marker.points.append(marker.points[0])

        self.pub_marker.publish(marker)

    def make_point(self, x, y):
        from geometry_msgs.msg import Point
        p = Point()
        p.x = float(x) * 0.001
        p.y = float(y) * 0.001
        p.z = 0.0
        return p


def main(args=None):
    rclpy.init(args=args)
    node = BeadVisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
