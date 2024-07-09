# Important Note: The aRMSD code is modified from the original aRMSD code. `acore.py` is modified to return corresponding atomic indices of reactant complex and product. 

![alt tag](./aRMSD_logo.png)

An open toolbox for structural comparison between two molecules with various capabilities to explore different aspects of structural similarity and diversity. Using data from common file formats the minimum RMSD is found by combining the functionalities of the Hungarian and the Kabsch algorithm. Crystallographic data provided from cif files is fully supported and the results can be rendered with the help of the [vtk] (http://www.vtk.org/) package.

# News & Updates
This page is currently under construction and more files will be added from day to day. So the best way to keep track of all changes and the ongoing delevopment process is by inspecting the [changelog] (./CHANGELOG.md). Please note that *aRMSD* was developed under Windows and is therefore not tested under other operating systems. Any feedback concerning the execution or PyInstaller compilation on Linux / Mac / ... is greatly appreciated.

The official publication was published recently and can be found under (DOI: 10.1021/acs.jcim.6b00516). If you use *aRMSD* in your work, please cite: 

**A. Wagner, H.-J. Himmel, J. Chem. Inf. Model, 2017, 57, 428-438.**

# Installation
*aRMSD* can be [installed] (./INSTALLATION.md) in two ways, either via pip (in this case it will be used as a Python module or Python application) or you can download the source code and compile it into a single standalone executable. In any case some packages are required which are listed below:

    * Python 2.7 or 3.6
    * numpy
    * vtk
    * matplotlib
    * PyQt4
    * uncertainties

optional:

    * Cython [performance improvements]
    * openbabel / pybel [additional file formats]

In order adjust the source code by yourself, always make sure to have the required Python pacakges installed and download the latest version of the master branch. Whenever possible it is recommended to add and update packages using pre-compiled Python [wheels] (http://www.lfd.uci.edu/~gohlke/pythonlibs/) suited for your operating system which can be installed via pip. If you add features or wish to have an idea implemented or a bug fixed, contact me or make a [request] (https://github.com/armsd/aRMSD/issues).

**a) Usage as Python application:**

- The easiest way to use *aRMSD*.

Simply clone the project or download the zip file from GitHub, navigate to the armsd folder and run aRMSD.py, e.g. from command line

```bash
python aRMSD.py
```

**b) Usage as Python module:**

- This will install *aRMSD* as a Python module that can be imported and used in other applications. This allows for both the most extensive utilization of the code and provides the possibility to keep an eye on the ongoing development via pip.

Install the program by typing

```bash
pip [to be added]
```

in your command line.

**c) Executable compiled with PyInstaller:**

- Produces a single file which can be copied anlongside the *settings.cfg* and the *xsf folder* to different machines with the same architecture. Once the program has been compiled, this is probably the easiest way to use *aRMSD* - especially for users that are unfamiliar with Python. 

First ensure that you have the latest version of [PyInstaller] (http://www.pyinstaller.org/) or install it with pip.

```bash
pip install pyinstaller
```

Download the current master branch of *aRMSD*, extract the files and navigate to the main folder. Run the compilation script in an interactive Python shell or from command line by typing

```bash
python compile_aRMSD.py
```

This will create a single executable file in the armsd folder and should work for all operating systems. Temporary files will be created during this process (the compilation will take around 30 min, depending on the machine) and deleted after the executable is created. Optional arguments can be given to make use of [Cython] (http://cython.org/) and [openbabel] (http://openbabel.org/wiki/Main_Page). Note that Cython C compiler should be specified if several options are available.

```bash
python compile_aRMSD.py --use_cython=True --cython_compiler=msvc --use_openbabel=True --overwrite=True
```

If you are using Python 3.6, there is a bug in the PyInstaller entry script and typing pyinstaller in a shell will not start a correct process. To fix this, go in the Python installation folder and edit the pyinstaller-script.py file: add quotes around the path in the first program line (e.g. "c:\program files\python36\python.exe")) 

# Documentation and Tutorial
To use the program, start the executable (if you are running aRMSD for the first time it is recommended to start the executable from command line to catch potential error messages) or run the application in a Python shell. Copy the example files to your current working directory and follow the instructions on screen. More information will be added in the near future.

# License
This package and its documentation are released under the [MIT License] (./LICENSE)
