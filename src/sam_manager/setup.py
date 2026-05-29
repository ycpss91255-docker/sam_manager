from setuptools import find_packages, setup

package_name = 'sam_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'structlog',
        'fastapi',
        'pillow',
        'python-multipart',
    ],
    zip_safe=True,
    maintainer='cyc',
    maintainer_email='ycpss91255@gmail.com',
    description='sam_manager Layer 4 — BackendInterface + MockBackend.',
    license='Apache-2.0',
    extras_require={'test': ['pytest', 'pytest-cov', 'httpx']},
    entry_points={
        'console_scripts': [],
    },
)
