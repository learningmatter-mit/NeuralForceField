# Neural Network code to perform energy, force, Hessian computations based on graph convolution 

## Getting Started 
First you need to set up your environment with the necessary python intall. This code is developed to be compatible with [graphbuilder](https://github.mit.edu/MLMat/graphbuilder) by William Harris. However the code is still under development, so please be sure to clone wujie branch to be compatible 

be sure to install the packages in the following way. 

### install required packages(It is advised that you create an new environment to intall alls the packages)

pytorch: ```conda install pytorch torchvision cudatoolkit=10.0 -c pytorch```
(Be sure to know your CUDA version and python version and install accordinly)
sk-learn: ```conda install scikit-learn```

if you want a install with database stuff, follow [Wil's tutorial](https://github.mit.edu/MLMat/mpnnet/blob/master/docs/README.md)


### clone Repos

set up directories. (You don't have to do this, just change the import paths when you run test in the notebook)

```bash
cd ~/
make Repo
cd Repo
mkdir projects
cd projects
```

clone graphbuilder (wujie branch)

```git clone --single-branch --branch wujie host:git@github.mit.edu:MLMat/graphbuilder.git```

clone NeuralForceField Code (master branch)
```git clone git@github.mit.edu:MLMat/NeuralForceField.git```

You will find two directories `NeuralForceField` and graphbuilder, and be sure to add them in your PYTHONPATH

##  Tutorial

You will find a tutorial.ipynb in the notebooks folder 



