from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow-gpu==1.1']

setup(
    name='trainer',
    version='1.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Sample convolutional network for recognising hand-written digits (MNIST dataset).'
)