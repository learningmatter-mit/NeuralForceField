# Neural Force Field

The Neural Force Field (NFF) code is an API based on SchNet [1-4], DimeNet [5], PaiNN [6] and DANN [7]. It provides an interface to train and evaluate neural networks for force fields. It can also be used as a property predictor that uses both 3D geometries and 2D graph information [8].

This code repository is developed in the Learning Matter Lab (led by prof. Rafael Gomez-Bombarelli) at MIT.

## Conda environment

We highly recommend creating a `conda` environment to run the code. To do that, use the following commands:

```bash
conda upgrade conda
conda create -n nff python=3.7.10 scikit-learn==0.24.1 pytorch=1.9.0 cudatoolkit=10.2 ase==3.21.1 pandas==1.2.2 pymatgen==2021.2.16 sympy==1.7.1 rdkit==2020.09.4 hyperopt==0.2.5 jq==1.6 openbabel==3.1.1 -c pytorch -c conda-forge -c rdkit -c openbabel
```

Next install remaining pip requirements:

```bash
conda activate nff
pip install sigopt==7.3.0  e3fp==1.2.3 ipykernel==5.5.0 performer-pytorch==1.0.11 nglview==3.0.1
```

To ensure that the `nff` environment is accessible through Jupyter, add the the `nff` display name:
```bash
python -m ipykernel install --user --name nff --display-name "Python [conda env:nff"]
```

## Installation

If you would like to install NFF as a package, you can do so by running

```bash
pip install .
```

Otherwise you can put NFF in your python path by adding the following lines to `~/.bashrc` (linux) or `~/.bash_profile` (mac):
```bash
export NFFDIR=<path to NFF>
export PYTHONPATH=$NFFDIR:$PYTHONPATH
```

This is useful if you'll be modifying the NFF code, because modifications in the download folder won't change anything in the conda directory where it's been installed. 


## Usage

### Jupyter notebooks 

#### Force field
A series of tutorials illustrating how `nff` can be used in conjunction with Jupyter Notebooks or other scripts is provided in the `tutorials/` folder. It also covers how to integrate a pre-trained model with an ASE calculator, how to perform ground state molecular dynamics (MD) and excited state non-adiabatic MD, and how to train different model types like DimeNet and PaiNN.

#### Property predictor
While `scripts/cp3d/README.md` explains in depth how to use the scripts, the notebook `07_cp3d.ipynb` goes into some detail about what happens behind the scenes. In this notebook you'll see how the datasets get made and what the models look like.

### Command line

#### Force field
The simplest way to use the `nff` package is to use the premade scripts (in the `scripts`) folder. As an example, to train a SchNet model with the default parameters using the example dataset (ethanol geometries) from the command line, run the command

```bash
nff_train.py train schnet tutorials/data/dataset.pth.tar $HOME/train_model --device cuda:0
```
This will use 60% of the dataset for training, 20% for validation and 20% for testing. The training will happen on the device `cuda:0`. Results of training, checkpoints and hyperparameters will be saved on the path `$HOME/train_model`.

#### Property predictor
NFF also contains modules that predict properties from 3D geometries of conformers. These include the SchNet model, expanded to include multiple conformers, as well as the ChemProp3D (CP3D)  model, which also includes graph information. A series of scripts for these modules can be found in `scripts/cp3d`. An in-depth discussion of how to use these scripts can be found in `scripts/cp3d/README.md`.   

### Pre-trained models
A set of pre-trained models can be found in `models`.

### Adversarial Attacks

NFF allows the usage of NN ensembles to perform uncertainty quantification and adversarial sampling of geometries. The complete tutorials on how to perform such analysis is available at the [Atomistic Adversarial Attacks repository](https://github.com/learningmatter-mit/Atomistic-Adversarial-Attacks), and the theory behind this differentiable sampling strategy is available at [our paper](https://www.nature.com/articles/s41467-021-25342-8) [9].

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

* [5] J. Klicpera, G. Janek, S. Günnemann. *Directional message passing for molecular graphs.* ICLR (2020). [URL](https://openreview.net/attachment?id=B1eWbxStPH&name=original_pdf).

* [6] K. T. Schütt, O. T. Unke, M. Gastegger. *Equivariant message passing for the prediction of tensorial properties and molecular spectra*. arXiv preprint, 2021. [arXiv:2102.03150](https://arxiv.org/pdf/2102.03150.pdf)

* [7] S. Axelrod, E. Shakhnovich, R. Gómez-Bombarelli. *Excited state, non-adiabatic dynamics of large photoswitchable molecules using a chemically transferable machine learning potential.* arXiv preprint (2021). [arXiv:2108.04879](https://arxiv.org/pdf/2108.04879.pdf)

* [8] S. Axelrod and R. Gomez-Bombarelli. *Molecular machine learning with conformer ensembles.* arXiv preprint (2020). [arXiv:2012.08452](https://arxiv.org/abs/2012.08452?fbclid=IwAR2KlinGWeEHTR99m8x9nu2caURqIg04nQkimqzYRcTIqFq6qgv6_RgmVzo).

* [9] D. Schwalbe-Koda, A.R. Tan, and R. Gomez-Bombarelli. *Differentiable sampling of molecular geometries with uncertainty-based adversarial attacks.* Nat. Commun. **12**, 5104 (2021). [URL](https://doi.org/10.1038/s41467-021-25342-8).

