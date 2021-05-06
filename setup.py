from setuptools import setup , Extension

setup(name='mykmeanssp',
      version='1.0',
      description='kmeans',
      ext_modules=[Extension('mykmeanssp', sources=['kmeans.c'])])