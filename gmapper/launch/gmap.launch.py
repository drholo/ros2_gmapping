from launch import LaunchDescription
from launch.substitutions import EnvironmentVariable
import launch.actions
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = launch.substitutions.LaunchConfiguration('use_sim_time', default='true')

    gmapper_node = Node(
        package='gmapper',
        executable='gmap',
        output='screen',
        parameters=[{'use_sim_time':use_sim_time}]
        )

    return LaunchDescription([
        gmapper_node
    ])