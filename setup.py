from setuptools import setup, find_packages

setup(
    name='lib',
    version='0.1.0',
    packages=find_packages(include=['lib', 'lib.*']),
    install_requires=[
        'Pyomo',
    ]
)
