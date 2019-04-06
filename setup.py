from setuptools import setup, find_packages

package_name = 'retinanet'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    py_modules=[],
    zip_safe=True,
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'scikit-image',
        'Pillow',
        'pyyaml',
        'cffi'
    ],
    author='Yann Henon',
    maintainer='Andreas Klintberg',
    description='RetinaNet implementation in PyTorch',
    license='Apache License, Version 2.0',
    test_suite='pytest'
)
