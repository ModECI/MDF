# Installation

## Requirements

Python >=3.7 is required. Support on Python 3.11 is limited, see [this issue](https://github.com/ModECI/MDF/issues/362).

## Installation using pip

Use pip to install the latest version of MDF (plus dependencies) from [PyPI](https://pypi.org/project/modeci-mdf):
```
pip install modeci_mdf
```

## Installation from source
To install the MDF package from [source](https://github.com/ModECI/MDF) and run it locally:

### 1) Create a virtual environment (e.g. called `mdf-env`)
```
pip install virtualenv
virtualenv mdf-env
```

### 2) Activate the virtual environment
```
source mdf-env/bin/activate
```

### 3) Clone this repository
```
git clone https://github.com/ModECI/MDF.git
```

### 4) Change to the directory
```
cd MDF
```

### 5) Install the package
```
pip install .
```

Alternatively, to install MDF plus all of the modules required for the export/import interfaces (e.g. PsyNeuLink, NeuroML):

```
pip install .[all]
```


## Additional dependencies

To generate generate Graph images in MDF you require Graphviz which uses dot.

```
pip install graphviz
```
To render the generated DOT source code, you also need to install [Graphviz](https://www.graphviz.org/) ([download page](https://www.graphviz.org/download/), [installation procedure for Windows](https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224).

Make sure that the directory containing the dot executable is on your system's PATH (sometimes done by the installer; setting PATH on [Linux](https://stackoverflow.com/questions/14637979/how-to-permanently-set-path-on-linux-unix), [Mac](https://stackoverflow.com/questions/22465332/setting-path-environment-variable-in-osx-permanently), and [Windows](https://www.computerhope.com/issues/ch000549.htm).



## Generating ModECI MDF documentation offline

The ModECI MDF Documentation can be found online [here](https://mdf.readthedocs.io/en/latest). If you are working on MDF documentation or you make changes to the documentation, it is good practice to see if it is working as expected before pushing to the GitHub repository.
Here is a walkthrough on how to generate the ModECI MDF documentation offline

## Requirements

**Python version-3.10 is ideally used for generating MDF documentation offline but if not working, use python version-3.9. The steps are the same except in creating a virtual environment.**

The documentation is generated using [Sphinx](https://www.sphinx-doc.org). Make is also required. For **Windows** installation of Make, see [here](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows). For **Mac** installation of Make, see [here](https://formulae.brew.sh/formula/make)



### 1) Create a virtual environment with python
```
# install virtual environment

pip install virtualenv

# create & activate virtual environment for python 3.9

python3.9 -m virtualenv venv39
venv39\Scripts\activate

# or create & activate virtual environment for python 3.10

python3.10 -m virtualenv venv310
venv310\Scripts\activate
```

### 2) Clone MDF repository from GitHub into your local machine
```
git clone https://github.com/ModECI/MDF.git
```

### 3) Change into the MDF directory
```
cd MDF
```

### 4) Install all MDF package into the virtual environment
```
pip install .[all]
```

### 5) Change directory into sphinx folder
```
# for windows
cd docs\sphinx

# for Mac/Linux
cd docs/sphinx
```

### 6) Create offline documentation in sphinx folder
```
# To allow a fresh start when making the documentation
make clean

# To make the documentation
make html
```


### 7) Change directory into html folder and run the documentation offline
```
# for Windows go into build\html folder and double click on the index.html file, or:

cd build\html
index.html

# for Mac, go into build/html folder and double click on the index.html file or:
cd build/html
open index.html
```

The documentation will open up in your browser automatically or right click on the file and open in any browser of your choice.
