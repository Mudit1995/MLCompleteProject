from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path, 'r') as f:
        requirements = f.readlines()
        # Remove newlines and clean up
        requirements = [req.strip() for req in requirements]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='MLProject',
    version='0.0.1',
    packages=find_packages(where='src'),  # Ensures packages are found in the 'src' directory
    package_dir={'': 'src'},  # Map root package to 'src'
    install_requires=get_requirements('requirments.txt'),
    author='Mudit',
    author_email='mudit.m.aggarwal@gmail.com',
    description='A brief description of your project',
)
