
import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
import launch_ros.actions

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

    # rviz2 =  Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     arguments=['-d', rviz_config_dir],
    #     output='screen')


    controller = Node(package='control', executable='controller.py', output='screen', remappings= [('/move_base_simple/goal', '/goal/pose')])
   

    # map file path for simple_map
    map_file_path = os.path.join(
        get_package_share_directory('planning'), 
        'maps',
        'simple_map.yaml'
    )


    rviz_config_path = os.path.join(
        get_package_share_directory('planning'),
        'rviz',
        'lab3_config.rviz')

    # map_server node
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename':map_file_path}]
    )

    # path planner client node
    dummy_client = Node(
        package='planning',
        executable='dummy_planner_client.py',
        name='dummy_planner_client',
        output='screen',
    )
    
    path_planner = Node(
        package='planning',
        executable='path_planner.py',
        name='path_planner',
        output='screen'
        )

    # start the node lifecyle manager
    lifecycle_manager = launch_ros.actions.Node(
        package='nav2_lifecycle_manager',
    
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': True},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    ) 

    # map to odom static transform
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        arguments=["0.5", "0.2", "0", "0.78", "0", "0", "odom", "map"],
        output='screen'
    )


    # odom to base_link transform, remove if running sim or real robot
    static_tf_fake_robot = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_odom_robot',
        arguments=["0.5", "0.5", "0", "0.", "0", "0", "odom", "base_link"],
        output='screen'
    )



    # launch rviz2
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]
    )

    return LaunchDescription([ # return all of the launch functions
        declare_use_sim_cmd,
        gz_sim,
        controller,
        map_server,
        path_planner,
        lifecycle_manager, 
        # static_tf_fake_robot,
        static_tf,
        rviz2,
        dummy_client
    ])
