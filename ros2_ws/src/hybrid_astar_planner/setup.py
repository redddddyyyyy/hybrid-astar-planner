from setuptools import setup, find_packages

package_name = 'hybrid_astar_planner'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/planner.launch.py']),
        ('share/' + package_name + '/config', ['config/planner_params.yaml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Rajeev Reddy',
    maintainer_email='rajeev@example.com',
    description='Hybrid A* path planner for non-holonomic vehicles',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = hybrid_astar_planner.planner_node:main',
            'obstacle_publisher = hybrid_astar_planner.obstacle_publisher:main',
        ],
    },
)
