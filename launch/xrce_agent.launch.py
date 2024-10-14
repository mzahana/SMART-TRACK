import os
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    port = LaunchConfiguration('port')
    port_launch_arg = DeclareLaunchArgument(
        'port',
        default_value='8888'
    )
   
    xrce_agent_process = ExecuteProcess(
            cmd=[[
                'MicroXRCEAgent udp4 -p ',
                port
            ]],
            shell=True
    )

    ld = LaunchDescription()

    ld.add_action(port_launch_arg)
    ld.add_action(xrce_agent_process)

    return ld