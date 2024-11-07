from setuptools import setup

package_name = 'panda_teleop_joy'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='github@enunezs.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'panda_teleop_control = panda_teleop_joy.panda_teleop_control:main',
            'send_static_goal = panda_teleop_joy.send_static_goal:main',
        ],
    },
)
