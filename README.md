# Neural Force Field
The Neural Force Field (NFF) code is an API based on SchNet [1-4], DimeNet [5], PaiNN [6-7], DANN [8], CHGNet [9], and MACE [10,11]. It provides an interface to train and evaluate neural networks (NNs) for force fields. It can also be used as a property predictor that uses both 3D geometries and 2D graph information [12]. NFF also allows the usage of NN ensembles to perform uncertainty quantification and adversarial sampling of geometries. The complete tutorials on how to perform such analysis is available at the [Atomistic Adversarial Attacks repository](https://github.com/learningmatter-mit/Atomistic-Adversarial-Attacks), and the theory behind this differentiable sampling strategy is available at [our paper](https://www.nature.com/articles/s41467-021-25342-8) [13].

This code repository is developed in the [Learning Matter Lab](http://gomezbombarelli.mit.edu) (led by Prof. Rafael Gomez-Bombarelli) at MIT.

## Conda environment
We highly recommend creating a `conda` environment to run the code. To do that, use the following command to create the `nff` conda environment:

```bash
conda upgrade conda
conda env create -f environment.yml
```

To ensure that the `nff` environment is accessible through Jupyter, add the the `nff` display name:
```bash
python -m ipykernel install --user --name nff --display-name "Python [conda env:nff"]
```

## Installation
Install Pytorch separately.
```bash
# Torch 2.4.0 with CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

If you would like to use NFF as a package (no development), you can do so by running the following command in the root directory of the repository:
```bash
pip install .
```

Use the `-e` flag if you want to develop NFF:
```bash
pip install -e '.[dev]'
pip install mace-torch=0.3.7 # install mace-torch with the right version this is temporary work around due to the UMA compatibility
```

## Usage
For those just getting started, we recommend referring to the [wiki](TBD) or advanced users can just jump right in. A Jupyter notebook interface or a command-line interface can be used with NFF. There is also a high-throughput option for using NFF, where you use HTVS to run simulations or perform calculations with NFF.

A set of pre-trained models can be found in [`models`](https://github.com/learningmatter-mit/NeuralForceField/tree/master/models). To take take an under-the-hood look at the architecture of all the models available with NFF, go to [`nff/nn/models`](https://github.com/learningmatter-mit/NeuralForceField/tree/master/nff/nn/models), and see the underlying modules with supporting functions in [`nff/nn/modules`](https://github.com/learningmatter-mit/NeuralForceField/tree/master/nff/nn/modules).

### Jupyter notebooks interface
Please refer to the [tutorials](https://github.com/learningmatter-mit/NeuralForceField/tree/master/tutorials) to see how to set up a Jupyter notebook interface with NFF.

### Command-line interface

#### Force field
The simplest way to use the `nff` package is to use the premade scripts (in the `scripts`) folder. As an example, to train a SchNet model with the default parameters using the example dataset (ethanol geometries) from the command line, run the command

```bash
nff_train.py train schnet tutorials/data/dataset.pth.tar $HOME/train_model --device cuda:0
```
This will use 60% of the dataset for training, 20% for validation and 20% for testing. The training will happen on the device `cuda:0`. Results of training, checkpoints and hyperparameters will be saved on the path `$HOME/train_model`.

#### Property predictor
NFF also contains modules that predict properties from 3D geometries of conformers. These include the SchNet model, expanded to include multiple conformers, as well as the ChemProp3D (CP3D)  model, which also includes graph information. A series of scripts for these modules can be found in `scripts/cp3d`. An in-depth discussion of how to use these scripts can be found in `scripts/cp3d/README.md`. 

### HTVS interface

Please refer to the [wiki](TBD/HTVS-interface-with-NFF) to see how to use NFF through HTVS.

### Adversarial Attacks

NFF allows the usage of NN ensembles to perform uncertainty quantification and adversarial sampling of geometries. The complete tutorials on how to perform such analysis is available at the [Atomistic Adversarial Attacks repository](https://github.com/learningmatter-mit/Atomistic-Adversarial-Attacks), and the theory behind this differentiable sampling strategy is available at [our paper](https://www.nature.com/articles/s41467-021-25342-8) [13].

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

* [7] S. Axelrod, E. Shakhnovich, R. Gómez-Bombarelli. *Thermal half-lives of azobenzene derivatives: virtual screening based on intersystem crossing using a machine learning potential.* arXiv preprint (2022). [arXiv:2207.11592](https://arxiv.org/abs/2207.11592).

* [8] S. Axelrod, E. Shakhnovich, R. Gómez-Bombarelli. *Excited state non-adiabatic dynamics of large photoswitchable molecules using a chemically transferable machine learning potential.* Nat. Commun. **13**, 3440 (2022). [URL](https://www.nature.com/articles/s41467-022-30999-w)

* [9] B. Deng, P. Zhong, K. Jun, J. Riebesell, K. Han, C. J. Bartel, G. Ceder. *CHGNet as a pretrained universal neural network potential for charge-informed atomstic modelling.* Nat. Mach. Intell. **5**, 1031 (2023). [10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3)

* [10] I. Batatia, D. P. Kovács, G. N. C. Simm, C. Ortner, G. Csányi. *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.* Advances in Neural Information Processing Systems 35, pp. 11423–11436 (2022). [link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html)

* [11] I. Batatia et al. *A foundation model for atomistic materials chemistry.* arXiv preprint, 2023. [2401.00096](https://arxiv.org/abs/2401.00096)

* [12] S. Axelrod and R. Gomez-Bombarelli. *Molecular machine learning with conformer ensembles.* arXiv preprint (2020). [arXiv:2012.08452](https://arxiv.org/abs/2012.08452?fbclid=IwAR2KlinGWeEHTR99m8x9nu2caURqIg04nQkimqzYRcTIqFq6qgv6_RgmVzo).

* [13] D. Schwalbe-Koda, A.R. Tan, and R. Gomez-Bombarelli. *Differentiable sampling of molecular geometries with uncertainty-based adversarial attacks.* Nat. Commun. **12**, 5104 (2021). [URL](https://doi.org/10.1038/s41467-021-25342-8).

