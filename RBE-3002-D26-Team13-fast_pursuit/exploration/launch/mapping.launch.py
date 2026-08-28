import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

from launch.conditions import UnlessCondition
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    # --- 1. Declare the Argument ---
    # This allows you to pass use_sim:=True in the terminal
    use_sim_arg = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Whether to use simulation (Gazebo) and sim_time'
    )

    # Reference the value of the argument
    use_sim_time = LaunchConfiguration('use_sim')

    # Paths
    exploration_dir = get_package_share_directory('exploration')
    planning_dir = get_package_share_directory('planning')
    simulation_dir = get_package_share_directory('simulation')
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')

    rviz_config_dir = os.path.join(planning_dir, 'rviz', 'lab3_config.rviz')
    
    # --- 2. Included Launch Files ---
    laser_filter = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='laser_filter',
        parameters=[
            os.path.join(exploration_dir, 'config', 'laser_filter_params.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('scan',          '/scan'),           # input: raw from driver
            ('scan_filtered', '/scan_filtered'),  # output: normalized
        ],
        condition=UnlessCondition(use_sim_time),  # only on real robot — sim scan is already consistent
        output='screen'
    )
    # SLAM Toolbox (Passing use_sim_time)
    my_config_path_real = os.path.join(exploration_dir, 'config', 'slam.yaml')      # scan_topic: /scan_filtered
    my_config_path_sim  = os.path.join(exploration_dir, 'config', 'slam_sim.yaml')  # scan_topic: /scan

    slam_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
        ),
        condition=IfCondition(use_sim_time),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'slam_params_file': my_config_path_sim   # ← sim config
        }.items()
    )

    slam_real = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
    ),
    condition=UnlessCondition(use_sim_time),
    launch_arguments={
        'use_sim_time': 'false',
        'slam_params_file': my_config_path_real
    }.items()
)

    # Simulation (Only runs IF use_sim is True)
    included_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(simulation_dir, 'launch', 'world_test.launch.py')
        ),
        condition=IfCondition(use_sim_time),
        launch_arguments={'world': 'small-field'}.items() 
    )

    # --- 3. Nodes (Passing use_sim_time to parameters) ---
    dummy_client = Node(
        package='planning',
        executable='dummy_planner_client.py',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    novelty = Node(
        package='exploration',
        executable='novelty_mapping_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # novelty = Node(
    #     package='exploration',
    #     executable='novelty_mapping.py',
    #     parameters=[{'use_sim_time': use_sim_time}],
    #     output='screen'
    # )

    path_planner = Node(
        package='planning', 
        executable='path_planner.py', 
        output='screen', 
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    controller = Node(
        package='control', 
        executable='controller.py', 
        output='screen', 
        remappings=[('/move_base_simple/goal', '/goal/pose')], 
        parameters=[{'use_sim_time': use_sim_time}]
    )
   

    # --- 4. Launch Description Construction ---
    ld = LaunchDescription()

    # Add the argument first
    ld.add_action(use_sim_arg)
    
    # Add actions/nodes
    ld.add_action(laser_filter)
    ld.add_action(included_simulation)
    ld.add_action(slam_real)
    ld.add_action(slam_sim)
    ld.add_action(controller)
    ld.add_action(rviz2)
    ld.add_action(novelty)
    ld.add_action(path_planner)
    ld.add_action(dummy_client)

    return ld