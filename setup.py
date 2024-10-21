from setuptools import setup
import os
from glob import glob

package_name = 'smart_track'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('rviz/*.rviz')),
        (os.path.join('share', package_name), glob('config/mavros/*.yaml')),
        (os.path.join('share', package_name), glob('config/kf/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mohamed Abdelkader',
    maintainer_email='mohamedashraf123@gmail.com',
    description='Drone-to-drone perception pipeline',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = smart_track.detection_node:main',
            'yolo2pose_node = smart_track.yolo2pose_node:main',
            'drone_marker_node = smart_track.drone_marker_node:main',
            'offboard_control = smart_track.offboard_control_node:main',
            'gt_target_tf = smart_track.gt_target_tf:main',
        ],
    },
)
