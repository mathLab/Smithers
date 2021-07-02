from setuptools import setup, Command, find_packages
import os
import sys

meta = {}
with open("smithers/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
NAME = meta['__title__']
DESCRIPTION = "Smithers' a Mathematical Interdisciplinary Toolbox for Helping Engineers Researchers and Scientist"
URL = 'https://github.com/mathLab/Smithers'
MAIL = meta['__mail__']
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS=''

REQUIRED = [
    'future', 'numpy', 'scipy',	'matplotlib',
]

EXTRAS = {
    'docs': ['Sphinx', 'sphinx_rtd_theme'],
    'vtk': ['vtk'],
}

LDESCRIPTION = (
    "Smithers is a generic library for scientific computing developed in "
    "Python that aims to facilitate the development of many typical routines. "
    "It is substantially a multi-purpose toolbox that inherits functionality "
    "from other packages to make easier and compact the coding of recurrent "
    "workflows."
)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
	classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
	],
	keywords=KEYWORDS,
	url=URL,
	license='MIT',
	packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    test_suite='nose.collector',
	tests_require=['nose'],
	include_package_data=True,
	zip_safe=False,
)
