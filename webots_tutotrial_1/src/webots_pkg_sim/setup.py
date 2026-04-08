from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'webots_pkg_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/sim_rover_launch.py']),
        ('share/' + package_name + '/urdf', ['urdf/new_rover.urdf']),
        ('share/' + package_name + '/worlds', ['worlds/my_world.wbt']),  # ← comma was missing here
        ('share/' + package_name + '/meshes', [
            'meshes/base_link.STL',
            'meshes/steering1_Link.STL',
            'meshes/steering2_Link.STL',
            'meshes/steering3_Link.STL',
            'meshes/steering4_Link.STL',
            'meshes/wheel1_Link.STL',
            'meshes/wheel2_Link.STL',
            'meshes/wheel3_Link.STL',
            'meshes/wheel4_Link.STL',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohanad',
    maintainer_email='eng.mohanadalexu@gmail.com',
    description='Webots ROS2 simulation package for Sojourner robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webots_driver = webots_pkg_sim.webots_driver:main',
            'rover_driver = webots_pkg_sim.rover_driver:main',
        ],
    },
)