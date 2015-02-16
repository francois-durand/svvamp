#!/usr/bin/env python
# -*- coding: utf-8 -*-


import setuptools
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'numpy', 'pandas', 'networkx', 'scipy', 'matplotlib'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='svvamp',
    version='0.0.2b2',
    description="Simulator of Various Voting Algorithms in Manipulating Populations",
    long_description=readme + '\n\n' + history,
    author="François Durand",
    author_email='fradurand@gmail.com',
    url='https://github.com/francois-durand/svvamp',
    packages=setuptools.find_packages(),
    package_dir={'svvamp':
                 'svvamp'},
    include_package_data=True,
    install_requires=requirements,
    license="GNU GPL 3",
    zip_safe=False,
    keywords='svvamp',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
