"""Packaging settings."""


from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

from setuptools import Command, find_packages, setup

from GPR1D import __version__


this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
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
    version = __version__,
    description = 'Classes for Gaussian Process Regression fitting of 1D data with errorbars.',
    long_description = long_description,
    url = 'https://gitlab.com/aaronkho/GPR1D.git',
    author = 'Aaron Ho',
    author_email = 'a.ho@differ.nl',
    license = 'MIT',
    classifiers = [
        'Intended Audience :: Developers',
        'Topic :: Utilities',
        'License :: MIT',
        'Natural Language :: English',
    ],
    keywords = 'gaussian processes, fitting, OMFIT',
    py_modules = ['GPR1D'],
    scripts = ['scripts/GPR1D_demo.py', 'guis/GPR1D_GUI.py'],
    install_requires = ['numpy', 'scipy'],
    extras_require = {
        'scripts': ['matplotlib'],
        'guis': ['matplotlib', 'PyQt4', 'PyQt5'],
        'test': ['coverage', 'pytest', 'pytest-cov'],
    },
    entry_points = {
        'console_scripts': [
            'GPR1D_demo=GPR1D_demo.py',
            'GPR1D_GUI=GPR1D_GUI.py'
        ],
    },
    cmdclass = {'test': RunTests},
)
