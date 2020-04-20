from distutils.core import setup, Extension

lemodule = Extension('lemodule', sources = ['lemodule.c'])

setup (name = 'Le',
       version = '0.2',
       description = 'Machine Learning Framework',
       ext_modules = [lemodule])
