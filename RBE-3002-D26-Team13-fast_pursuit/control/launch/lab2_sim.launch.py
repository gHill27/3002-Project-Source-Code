import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart


def generate_launch_description():   

    use_sim_arg = LaunchConfiguration('use_sim')
    declare_use_sim_cmd = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Whether to start Gazebo simulation'
    )


    rviz_config_dir = os.path.join(
        get_package_share_directory('control'),
        'rviz',
        'control_pkg.rviz')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('turtlebot3_gazebo'), 'launch'), '/empty_world.launch.py']),
        launch_arguments = [
            ('use_sim_time', 'True')
        ], 
        condition=IfCondition(use_sim_arg)
    )

    rviz2 =  Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        output='screen')

    static_transform = Node(package = "tf2_ros", 
                       executable = "static_transform_publisher",
                       arguments=["0.5", "0.2", "0", "0.78", "0", "0", "odom", "map"] 
            )


    controller = Node(package='control', executable='controller.py', output='screen')
    path_generator = Node(package='control', executable='path_generator.py', output='screen')

    delayed_path_generator = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller,
            on_start=[
                TimerAction(
                    period=4.0,
                    actions=[path_generator]),
                ],
        ))

    return LaunchDescription([
        declare_use_sim_cmd,
        gz_sim,
        rviz2,
        static_transform,
        controller,
        delayed_path_generator #waits until controller is setup for consistency. 
        
    ])
