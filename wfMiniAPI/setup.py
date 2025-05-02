#!/usr/bin/env python3

__author__    = 'RADICAL-Cybertools Team, Tianle Wang, Ozgur Kilic'
__email__     = 'info@radical-cybertools.org'
__copyright__ = 'Copyright 2022-25, The RADICAL-Cybertools Team'
__license__   = 'MIT'


''' Setup script, only usable via pip. '''

import os

import subprocess as sp

from glob       import glob
from setuptools import setup, Command, find_packages


# ------------------------------------------------------------------------------
#
repo     = 'workflow-mini-apps'
name     = 'wfminiapps'
mod_root = 'src/%s/'    % name

root     = os.path.dirname(__file__) or '.'
readme   = open('%s/README.md' % root, encoding='utf-8').read()
descr    = 'An open source library that is used to make implementing emulated' \
           ' task in workflow mini-app simple. It support both Python and C++' \
           ' (OpenMP) backend and is targetting various different'             \
           ' architecture including CPU, NVIDIA GPU, AMD GPU and Intel GPU'
keywords = ['radical', 'cybertools', 'mini-app']


# ------------------------------------------------------------------------------
# get version info
version = open('%s/VERSION' % root).read().strip()


# ------------------------------------------------------------------------------
#
class RunTwine(Command):
    user_options = []
    def initialize_options(self): pass
    def finalize_options(self):   pass
    def run(self):
        _, _, _ret = sh_callout('python3 setup.py sdist upload -r pypi')
        raise SystemExit(_ret)


# ------------------------------------------------------------------------------
#
with open('%s/requirements.txt' % root, encoding='utf-8') as freq:
    requirements = freq.readlines()

with open('%s/requirements-gpu.txt' % root, encoding='utf-8') as freq:
    requirements_gpu = freq.readlines()



# ------------------------------------------------------------------------------
#
setup_args = {
    'name'               : name,
    'version'            : version,
    'description'        : descr,
    'long_description'   : readme,
    'long_description_content_type' : 'text/markdown',
    'author'             : __author__,
    'author_email'       : __email__,
    'maintainer'         : 'The RADICAL Group',
    'maintainer_email'   : 'radical@rutgers.edu',
    'url'                : 'http://radical-cybertools.github.io/%s/' % repo,
    'project_urls'       : {
        'Documentation': 'https://%s.readthedocs.io/en/latest/'      % name,
        'Source'       : 'https://github.com/radical-cybertools/%s/' % repo,
        'Issues' : 'https://github.com/radical-cybertools/%s/issues' % repo,
    },
    'license'            : 'MIT',
    'keywords'           : keywords,
    'python_requires'    : '>=3.8',
    'classifiers'        : [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
        'Topic :: System :: Distributed Computing',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix'
    ],
    'packages'           : find_packages('src'),
    'package_dir'        : {'': 'src'},
    'scripts'            : list(glob('mini-apps/*/*.py')),

    'package_data'       : {'': ['*.txt', '*.sh', '*.json', '*.gz', '*.c',
                                 '*.md', 'VERSION']},
    'install_requires'   : requirements,
    'extras_require'     : {'gpu': ['cupy']},
    'zip_safe'           : False,
    'cmdclass'           : {'upload': RunTwine},
}


# ------------------------------------------------------------------------------
#
setup(**setup_args)


# ------------------------------------------------------------------------------
# clean temporary files from source tree
os.system('rm -vrf src/%s.egg-info' % name)


# ------------------------------------------------------------------------------

