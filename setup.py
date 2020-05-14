#!/usr/bin/env python
from setuptools import setup, find_packages
from pathlib import Path

readme = Path('README.md').read_text()
requirements = Path('requirements.txt').read_text().split('\n')
__version__ = Path('version').read_text().strip()

setup(
    name='neuralDX7',
    version=__version__,
    author='Nintorac',
    author_email='nintoracaudio@gmail.com',
    url='https://github.com/nintorac/agoge',
    description='Models related to the DX7 fm synthesizer',
    long_description=readme,
    zip_safe=True,
    install_requires=requirements,
)