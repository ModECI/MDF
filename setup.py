#!/usr/bin/env python

from distutils.core import setup


import modeci_mdf
version = modeci_mdf.__version__

setup(name='modeci-mdf',
      version=version,
      description='ModECI (Model Exchange and Convergence Initiative) Model Description Format',
      author='Padraig Gleeson; ...',
      author_email='p.gleeson@gmail.com',
      url='https://www.modeci.org',
      packages=['modeci_mdf'],
      install_requires=[
        'neuromllite>=0.1.9',
        'matplotlib'],
      classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering']
     )
