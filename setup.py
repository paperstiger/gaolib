from distutils.core import setup
from pkgutil import walk_packages

import gaolib

def find_packages(path, prefix=""):
    yield prefix
    prefix = prefix + "."
    for _, name, ispkg in walk_packages(path, prefix):
        if ispkg:
            yield name

setup(
        name='gaolib',
        version='0.1.1',
        author='Gao Tang',
        author_email='gaotang2@illinois.edu',
        packages=list(find_packages(gaolib.__path__, gaolib.__name__)),
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
            'sympy>=1.0.0',
            'sklearn',
            'numba'
        ],
)
