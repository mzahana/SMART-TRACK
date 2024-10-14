import os
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    PX4_DIR = os.getenv('PX4_DIR')

    if PX4_DIR is not None:
        print(f'The value of PX4_DIR is {PX4_DIR}')
    else:
        print('PX4_DIR is not set')
        sys.exit(1)

    namespace = LaunchConfiguration('gz_ns')
    namespace_launch_arg = DeclareLaunchArgument(
        'gz_ns',
        default_value=''
    )

    headless = LaunchConfiguration('headless')
    headless_launch_arg = DeclareLaunchArgument(
        'headless',
        default_value='0'
    )

    gz_world = LaunchConfiguration('gz_world')
    gz_world_launch_arg = DeclareLaunchArgument(
        'gz_world',
        default_value='default'
    )

    gz_model_name = LaunchConfiguration('gz_model_name')
    gz_model_name_launch_arg = DeclareLaunchArgument(
        'gz_model_name',
        default_value='x500'
    )

    px4_autostart_id = LaunchConfiguration('px4_autostart_id')
    px4_autostart_id_launch_arg = DeclareLaunchArgument(
        'px4_autostart_id',
        default_value='4001'
    )

    instance_id = LaunchConfiguration('instance_id')
    instance_id_launch_arg = DeclareLaunchArgument(
        'instance_id',
        default_value='0'
    )

    xpos = LaunchConfiguration('xpos')
    xpos_launch_arg = DeclareLaunchArgument(
        'xpos',
        default_value='0.0'
    )

    ypos = LaunchConfiguration('ypos')
    ypos_launch_arg = DeclareLaunchArgument(
        'ypos',
        default_value='0.0'
    )

    zpos = LaunchConfiguration('zpos')
    zpos_launch_arg = DeclareLaunchArgument(
        'zpos',
        default_value='0.2'
    )

    cmd1_str="cd {} && ".format(PX4_DIR)
    cmd2_str="PX4_SYS_AUTOSTART={} PX4_GZ_MODEL={} PX4_MICRODDS_NS={} PX4_GZ_MODEL_POSE='{},{},{}' ./build/px4_sitl_default/bin/px4 -i {}".format(px4_autostart_id, gz_model_name, namespace, xpos,ypos,zpos, instance_id)
    cmd_str = cmd1_str+cmd2_str
    px4_sim_process = ExecuteProcess(
        cmd=[[
            'cd ',PX4_DIR ,' && ',
            'PX4_SYS_AUTOSTART=', px4_autostart_id,
            ' PX4_GZ_MODEL=', gz_model_name,
            ' PX4_UXRCE_DDS_NS=',namespace,
            " PX4_GZ_MODEL_POSE='",xpos,',',ypos,',',zpos,"'",
            ' PX4_GZ_WORLD=', gz_world,
            ' ./build/px4_sitl_default/bin/px4 -i ', instance_id
        ]],
        shell=True
    )

    ld = LaunchDescription()

    ld.add_action(headless_launch_arg)
    ld.add_action(gz_world_launch_arg)
    ld.add_action(gz_model_name_launch_arg)
    ld.add_action(px4_autostart_id_launch_arg)
    ld.add_action(instance_id_launch_arg)
    ld.add_action(xpos_launch_arg)
    ld.add_action(ypos_launch_arg)
    ld.add_action(zpos_launch_arg)
    ld.add_action(namespace_launch_arg)
    ld.add_action(px4_sim_process)

    return ld