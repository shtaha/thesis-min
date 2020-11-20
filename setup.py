from setuptools import setup, find_packages

setup(
    name='MIP_oracle',
    packages=find_packages(include=['MIP_oracle', 'MIP_oracle.*']),
    install_requires=[
        'Pyomo',
    ]
)
