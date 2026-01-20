from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    base = "/home/ho/BEADtrain/BEADtrain"

    return LaunchDescription([
        Node(
            executable="python3",
            arguments=[f"{base}/bead_unet_node.py"],
            output="screen",
        ),
        Node(
            executable="python3",
            arguments=[f"{base}/bead_visualizer_node.py"],
            output="screen",
        ),
        Node(
            executable="python3",
            arguments=[f"{base}/bead_grind_check_node.py"],
            output="screen",
        ),
    ])
