import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, ExecuteProcess,
    IncludeLaunchDescription, LogInfo, RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def wait_for(command):
    return ExecuteProcess(
        cmd=[
            'bash', '-c',
            f'for i in $(seq 1 60); do {command} && exit 0; '
            'sleep 2; done; exit 1',
        ],
        output='screen',
    )


def then_start(check, actions, label):
    def on_exit(event, context):
        if event.returncode != 0:
            return [EmitEvent(event=Shutdown(
                reason=f'Timed out waiting for {label}'))]
        return [LogInfo(msg=f'{label} ready'), *actions]

    return RegisterEventHandler(
        OnProcessExit(target_action=check, on_exit=on_exit))


def generate_launch_description():
    share = get_package_share_directory('articubot_one')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(
            share, 'worlds', 'small_test_room.sdf'),
    )
    explorer_arg = DeclareLaunchArgument(
        'start_explorer', default_value='true')
    gazebo_arg = DeclareLaunchArgument(
        'show_gazebo', default_value='true')
    rviz_arg = DeclareLaunchArgument(
        'show_rviz', default_value='true')

    def include(filename, arguments=None):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, 'launch', filename)),
            launch_arguments=(arguments or {}).items(),
        )

    sim = include('launch_sim.launch.py', {
        'world': LaunchConfiguration('world'),
    })
    slam = include('online_async_launch.py')
    nav2 = include('navigation_launch.py', {
        'use_sim_time': 'true',
        'params_file': os.path.join(
            share, 'config', 'nav2_params.yaml'),
    })

    gazebo_gui = ExecuteProcess(
        cmd=['ign', 'gazebo', '-g'],
        condition=IfCondition(LaunchConfiguration('show_gazebo')),
        output='screen',
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(share, 'rviz', 'exploration.rviz')],
        condition=IfCondition(LaunchConfiguration('show_rviz')),
        output='screen',
    )
    explorer = Node(
        package='articubot_one',
        executable='frontier_explorer.py',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('start_explorer')),
        output='screen',
    )

    scan_ready = wait_for(
        'timeout 6 ros2 topic echo /scan --once --field range_max '
        '>/dev/null 2>&1')
    map_ready = wait_for(
        'timeout 8 ros2 topic echo /map --once --field info '
        '>/dev/null 2>&1')
    nav_ready = wait_for(
        "timeout 6 ros2 lifecycle get /bt_navigator 2>/dev/null "
        "| grep -q 'active \\[3\\]'")

    return LaunchDescription([
        world_arg, explorer_arg, gazebo_arg, rviz_arg,
        then_start(scan_ready, [slam, map_ready, gazebo_gui], '/scan'),
        then_start(map_ready, [nav2, nav_ready, rviz], '/map'),
        then_start(nav_ready, [explorer], 'Nav2'),
        sim,
        scan_ready,
    ])
