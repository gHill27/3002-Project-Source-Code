import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():

    map_file_path = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'final_lab_map105.946601094.yaml')
    rviz_config_path = os.path.join(get_package_share_directory('planning'), 'rviz', 'lab3_config.rviz')
    amcl_config_path = os.path.join(get_package_share_directory('exploration'), 'config', 'amcl.yaml')

    use_sim_arg = LaunchConfiguration('use_sim')
    declare_use_sim_cmd = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Whether to start Gazebo simulation'
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('simulation'), 'launch'), '/world_test.launch.py']),
        launch_arguments = [
            ('use_sim_time', 'True')
        ], 
        condition=IfCondition(use_sim_arg)
    )

    relocalize_call = TimerAction(
        period=5.0,  # wait 5 seconds before calling
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'service', 'call', '/reinitialize_global_localization',
                    'std_srvs/srv/Empty', '{}'],
                output='screen'
            )
        ]
    )

    # 1. Map Server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': map_file_path}]
    )

    # 2. AMCL Node
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        parameters=[amcl_config_path]
    )

    # 3. Lifecycle Manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['map_server', 'amcl']
        }]
    )

    # 4. Rviz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    # 5. Controller
    controller = Node(
        package='control',
        executable='controller.py',
        output='screen', 
        remappings= [('/move_base_simple/goal', '/goal/pose')]
    )

    # 6. Localizer
    localizer = Node(
        package='exploration',
        executable='localizer.py',
        output='screen'
    )

    # 7. Path planner client
    planner_client = Node(
        package='planning',
        executable='dummy_planner_client.py',
        name='dummy_planner_client',
        output='screen'
    )
    
    # 8. Path planner
    path_planner = Node(
        package='planning',
        executable='path_planner.py',
        name='path_planner',
        output='screen'
    )

    return LaunchDescription([

        map_server,
        amcl,
        lifecycle_manager,
        rviz,
        controller,
        localizer,
        planner_client,
        path_planner,
        declare_use_sim_cmd,
        gz_sim,
        relocalize_call

    ])

    