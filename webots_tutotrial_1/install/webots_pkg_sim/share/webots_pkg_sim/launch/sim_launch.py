import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController

def generate_launch_description():
    package_dir = get_package_share_directory('webots_pkg_sim')
    robot_description_path = os.path.join(package_dir, 'urdf', 'robot.urdf')

    with open(robot_description_path, 'r') as f:
        robot_description = f.read()

    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', 'empty.wbt')
    )

    robot_driver = WebotsController(
        robot_name='rosbot_xl',
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    return LaunchDescription([
        webots,
        robot_driver,
        robot_state_publisher,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])