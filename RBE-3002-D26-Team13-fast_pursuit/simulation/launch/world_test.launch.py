import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    # base file path for the package
    sim_path = os.path.join(get_package_share_directory('simulation'))
    tb3_gz_pkg = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    # secondary file paths for locating resources
    models_path = os.path.join(sim_path, 'models')
    worlds_path = os.path.join(sim_path, 'worlds')

    # set gz sim resource path
    gz_sim_resource = SetEnvironmentVariable(
        name = 'GZ_SIM_RESOURCE_PATH',
        value=f"{models_path}:{worlds_path}:{sim_path}"
    )

    # arguments for gz sim
    arguments = LaunchDescription([
            DeclareLaunchArgument('world', default_value='small-field', description='small field for 3002.'),
    ])

    # actually run gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch'), '/gz_sim.launch.py']),
        launch_arguments = [
            ('gz_args', [LaunchConfiguration('world'),'.sdf',' -v4',' -r']),
            ('use_sim_time', 'True')
        ]
    )

    # spawn in turtlebot
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz_pkg, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': '0.15',
            'y_pose': '0.0',
            'use_sim_time': 'true'
        }.items()
    )
    # robot state publisher
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz_pkg, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )
    
    # launch each item defined above by returning the variable
    return LaunchDescription([
        gz_sim_resource,
        arguments,
        gazebo,
        spawn_turtlebot_cmd,
        robot_state_publisher_cmd
    ])