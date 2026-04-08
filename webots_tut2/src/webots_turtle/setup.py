from setuptools import find_packages, setup

package_name = 'webots_turtle'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', ['urdf/robot.urdf']),
        ('share/' + package_name + '/worlds', ['worlds/Webots_world.wbt']),
        ('share/' + package_name + '/launch', ['launch/sim_launch_2.py']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohanad',
    maintainer_email='eng.mohanadalexu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'webots_turtle = webots_turtle.driver_turtle:main',
        ],
    },
)
