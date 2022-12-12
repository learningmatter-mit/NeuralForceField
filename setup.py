import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="nff",
    version="1.0.0",
    author="Wujie Wang, Daniel Schwalbe-Koda, Simon Axelrod, Shi Jun Ang",
    email="{wwj,dskoda,saxelrod}@mit.edu",
    url="https://github.com/learningmatter-mit/NeuralForceField",
    packages=find_packages("."),
    scripts=[
        "scripts/nff_train.py",
    ],
    python_requires=">=3.5",
    package_data={'nff': ['utils/table_data/c6ab.npy',
                          'utils/table_data/functional_params.json',
                          'utils/table_data/r2r4.npy',
                          'utils/table_data/rcov.npy']},
    include_package_data=True,
    install_requires=[
#        "pytorch>=1.4.0",
#        "numpy",
#        "ase>=3.19.1",
#        "scikit-learn>=0.23.1",
#        "pandas>=1.0.5",
#        "networkx>=2.4",
#        "pymatgen>=2020.7.3",
#        "sympy>=1.6.1",
    ],
    license="MIT",
    description="Neural Force Field based on SchNet",
    long_description=read("README.md"),
)
