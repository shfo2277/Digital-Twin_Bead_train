#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from rclpy.time import Time

class CameraTestNode(Node):
    def __init__(self):
        super().__init__('camera_test_node')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, '/camera_forward_point', 10)
        self.timer = self.create_timer(0.5, self.timer_cb)
        self.camera_frame = 'camera_color_optical_frame'  # 네가 쓰는 카메라 프레임명으로

    def timer_cb(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link',
                self.camera_frame,
                Time()
            )
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return

        ps = PoseStamped()
        ps.header.frame_id = self.camera_frame
        ps.header.stamp = self.get_clock().now().to_msg()

        # ✅ 카메라 앞 0.5m (optical frame 기준 +Z 방향)
        ps.pose.position.x = 0.0
        ps.pose.position.y = 0.0
        ps.pose.position.z = 0.5
        ps.pose.orientation.w = 1.0

        ps_base = do_transform_pose(ps, trans)
        ps_base.header.frame_id = 'base_link'

        self.pub.publish(ps_base)
        self.get_logger().info(
            f"Camera forward point in base_link: "
            f"({ps_base.pose.position.x:.3f}, "
            f"{ps_base.pose.position.y:.3f}, "
            f"{ps_base.pose.position.z:.3f})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = CameraTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
