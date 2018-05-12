from distutils.core import setup
from pkgutil import walk_packages

import pyLib

def find_packages(path, prefix=""):
    yield prefix
    prefix = prefix + "."
    for _, name, ispkg in walk_packages(path, prefix):
        if ispkg:
            yield name

setup(
        name='pyLib',
        version='0.1.1',
        author='Gao Tang',
        author_email='gao.tang@duke.edu',
        packages=list(find_packages(pyLib.__path__, pyLib.__name__)),
        scripts=[],
        url='',
        license='LICENSE.txt',
        description='Useful tools for research',
        long_description=open('README.md').read(),
        install_requires=[
            'numpy>=1.13.0',
            'control>=0.7.0',
            'scipy>=1.0.0',
            'matplotlib>=1.13.0',
            'sympy>=1.5.0'
        ],
)
