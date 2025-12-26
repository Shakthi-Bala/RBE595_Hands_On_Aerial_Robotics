# setup.py
from setuptools import setup, Extension

ext = Extension(
    name="control",          # must match PyInit_control
    sources=["control.c"],   # your existing C file
)

setup(
    name="control",
    version="0.1.0",
    ext_modules=[ext],
)
