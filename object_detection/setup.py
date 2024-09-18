from setuptools import find_packages, setup

package_name = 'object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='athena',
    maintainer_email='athena@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "object_detection=object_detection.object_detection:main",
            "object_detection_trans=object_detection.object_detection_trans:main",
            "object_detection_transformed=object_detection.object_detection_transformed:main",
            "object_detection_opti=object_detection.object_detection_opti:main",
            "image_detection=object_detection.image_detection:main",
        ],
    },
)
