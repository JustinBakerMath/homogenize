from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
name='homogenize',
version='0.1',
description='Rank 1 Laminate Method for Solving Coefficient Diffusion Equation.',
author='Justin Baker',
author_email='baker@math.utah.edu',
packages=['homogenize'],  #same as name
install_requires=['pandas', 'numpy', 'matplotlib'], #external packages as dependencies
data_files=[('./out/', [])]
)
