import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController

def generate_launch_description():
    pkg  = get_package_share_directory('webots_pkg_sim')
    urdf = os.path.join(pkg, 'urdf', 'new_rover.urdf')

    with open(urdf, 'r') as f:
        robot_description = f.read()

    webots = WebotsLauncher(
        world=os.path.join(pkg, 'worlds', 'my_world.wbt')
    )

    driver = WebotsController(
        robot_name='new_rover',
        parameters=[{'robot_description': urdf}]
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    return LaunchDescription([
        webots,
        driver,
        rsp,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(
                    event=launch.events.Shutdown())]
            )
        )
    ])