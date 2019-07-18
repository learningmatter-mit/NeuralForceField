import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="nff",
    version="1.0.0",
    author="Wujie Wang, Daniel Schwalbe-Koda",
    email="{wwj,dskoda}@mit.edu",
    url="https://github.com/atomistic-machine-learning/schnetpack",
    packages=find_packages("."),
    scripts=[
        "scripts/nff_train.py",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=0.4.1",
        "numpy",
        "ase>=3.16",
        "tensorboardX",
    ],
    license="MIT",
    description="Neural Force Field based on SchNet",
    long_description=read("README.md"),
)