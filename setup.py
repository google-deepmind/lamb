# Copyright 2018 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Setup for pip package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['absl-py', 'numpy', 'dm-sonnet', 'six']
EXTRA_PACKAGES = {
    'tensorflow': ['tensorflow>=1.15.0', 'tensorflow-probability>=0.4.0'],
    'tensorflow with gpu': ['tensorflow-gpu>=1.8.0',
                            'tensorflow-probability-gpu>=0.4.0'],
}


setup(
    name='lamb',
    version='1.0',
    description=('LAnguage Modelling Benchmarks is '
                 'to tune and test Tensorflow LM models.'),
    long_description='',
    url='http://github.com/deepmind/lamb/',
    author='Gabor Melis',
    author_email='melisgl@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    zip_safe=False,
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
    keywords='lamb tensorflow language modelling machine learning',
)
