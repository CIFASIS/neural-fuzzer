#!/usr/bin/python2
from setuptools import setup

setup(
    name='Neural-Fuzzer',
    version='0.1',
    license='GPL3',
    description='',
    long_description="",
    url='http://cifasis.github.io/neural-fuzzer/',
    author='G.Grieco',
    author_email='gg@cifasis-conicet.gov.ar',
    scripts=['neural-fuzzer.py'],
    install_requires=[
        "keras",
        "h5py"
    ],
)

