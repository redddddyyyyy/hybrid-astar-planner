"""Launch file for Hybrid A* planner."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('hybrid_astar_planner')

    # Paths
    default_params = os.path.join(pkg_dir, 'config', 'planner_params.yaml')

    # Launch arguments
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to planner parameters file'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),

        # Planner node
        Node(
            package='hybrid_astar_planner',
            executable='planner_node',
            name='hybrid_astar_planner',
            output='screen',
            parameters=[
                params_file,
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/goal_pose', '/goal_pose'),
                ('/initialpose', '/initialpose'),
                ('/map', '/map'),
                ('/planned_path', '/planned_path'),
            ]
        ),
    ])
