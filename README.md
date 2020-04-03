# Neural Force Field

The Neural Force Field (NFF) code is an API based on SchNet [1-4]. It provides an interface to train and evaluate neural networks for force fields.

## Installation from source

This software requires the following packages:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [ase](https://wiki.fysik.dtu.dk/ase/)
- [networkx](https://networkx.github.io/)
- [sigopt](https://sigopt.com/)

For interfacing NFF with [ChemProp](https://github.com/chemprop/chemprop) [5], a module for property prediction based on 2D molecular graphs, the following additional packages are required:
- [gunicorn](https://gunicorn.org/)
- [RDKit](https://www.rdkit.org/)
- [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
- [flask](https://flask.palletsprojects.com/en/1.1.x/)
- [hyperopt](https://github.com/hyperopt/hyperopt)
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [tensorflow](https://www.tensorflow.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm)
- [munch](https://github.com/Infinidat/munch)
- [descriptasorus](https://github.com/bp-kelley/descriptastorus)
- [chemprop](https://github.com/chemprop/chemprop)

We highly recommend to create a `conda` environment to run the code. To do that, use the following commands:

```bash
conda upgrade conda
conda create -n nff python=3.7 scikit-learn pytorch=1.2.0 cudatoolkit=10.0 ase pandas \
pymatgen gunicorn rdkit torchvision flask hyperopt numpy scipy tensorflow tensorboardX \
tqdm -c pytorch -c conda-forge -c rdkit
```

Next install remaining pip requirements:

```bash
pip install munch sigopt git+https://github.com/bp-kelley/descriptastorus \ 
git+https://github.com/simonaxelrod/chemprop
```

You need to activate the `nff` environment to install the NFF package:

```bash
conda activate nff
```

Finally, install the `nff` package by running:

```bash
pip install .
```

## Usage

### Command line
The simplest way to use the `nff` package is to use the premade scripts (in the `scripts`) folder. As an example, to train a SchNet model with the default parameters using the example dataset (ethanol geometries) from the command line, run the command

```bash
nff_train.py train schnet tutorials/data/dataset.pth.tar $HOME/train_model --device cuda:0
```

This will use 60% of the dataset for training, 20% for validation and 20% for testing. The training will happen on the device `cuda:0`. Results of training, checkpoints and hyperparameters will be saved on the path `$HOME/train_model`.

### Usage with Jupyter Notebooks and other scripts

A series of tutorials illustrating how `nff` can be used in conjunction with Jupyter Notebooks or other scripts is provided in the `tutorials/` folder. It also covers how to integrate a pre-trained model with an ASE calculator.


## References

* [1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.  
*Quantum-chemical insights from deep tensor neural networks.*
Nature Communications **8**. 13890 (2017)   
[10.1038/ncomms13890](http://dx.doi.org/10.1038/ncomms13890)

* [2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)

* [3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet - a deep learning architecture for molecules and materials.* 
The Journal of Chemical Physics 148(24), 241722 (2018) [10.1063/1.5019779](https://doi.org/10.1063/1.5019779)

* [4] K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller.
*SchNetPack: A Deep Learning Toolbox For Atomistic Systems.*
J. Chem. Theory Comput. **15**(1), 448-455 (2019). [10.1021/acs.jctc.8b00908](https://doi.org/10.1021/acs.jctc.8b00908)

* [5] K. Yang, *et. al*. *Analyzing learned molecular representations for property prediction.*
J. Chem. Info. Model. **59**(8), 3370-3388 (2019). [10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237)




