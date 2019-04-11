"""Packaging settings."""


from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

from setuptools import Command, find_packages, setup


this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.rst'), encoding='utf-8') as file:
    long_description = file.read()


class RunTests(Command):
    """Run all tests."""
    description = 'run tests'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run all tests!"""
        errno = call(['py.test', '--cov=GPR1D.py', '--cov-report=term-missing',
                      '--ignore=lib/'])
        raise SystemExit(errno)


setup(
    name = 'GPR1D',
    version = '1.2.0',
    description = 'Classes for Gaussian Process Regression fitting of 1D data with errorbars.',
    long_description = long_description,
    url = 'https://gitlab.com/aaronkho/GPR1D.git',
    author = 'Aaron Ho',
    author_email = 'a.ho@differ.nl',
    license = 'MIT',
    classifiers = [
        'Intended Audience :: Developers',
        'Topic :: Utilities',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5'
    ],
    keywords = 'gaussian process regression, 1D data fitting, regression analysis, kriging',
    py_modules = ['GPR1D'],
    scripts = ['scripts/GPR1D_demo.py', 'guis/GPR1D_GUI.py'],
    install_requires = ['numpy>=1.13', 'scipy>=0.17'],
    extras_require = {
        'scripts': ['matplotlib'],
        'guis': ['matplotlib', 'PyQt4'],
        'test': ['coverage', 'pytest', 'pytest-cov'],
    },
    cmdclass = {'test': RunTests},
)
