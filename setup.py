#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='trace_clustering',
      version='1.0',
      description='Python package for clustering traces (sequences of data characterized by one or more features).',
      url='https://github.com/SRI-AIC/trace-clustering',
      author='Pedro Sequeira',
      author_email='pedro.sequeira@sri.com',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'plotly',
          'scikit-learn',
          'tqdm',
          'fastdtw'
      ],
      zip_safe=True)
