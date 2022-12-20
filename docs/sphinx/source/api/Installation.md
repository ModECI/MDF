# Installation

## Requirements

Python (>=3.7)

## Quick start

```
pip install modeci_mdf
```

## Installation from source
To install the MDF package from source and run it locally:

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



# Generating ModECI MDF documentation offline

The ModECI MDF Documentation can be found online [here](https://mdf.readthedocs.io/en/latest). If you are working on MDF documentation or you make changes to the documentation, it is good practice to see if it is working as expected before pushing to the Github repository.
Here is a walkthrough on how to generate the ModECI MDF documentation offline

## Requirements 

Python (3.9.0)

See Installation [here](https://www.python.org/downloads/release/python-390)

Add python version-3.9.0 to path

### 1). Create a virtual environment with python 3.9.0
```
# install virtual environment

pip install virtualenv

# create virtual environment using python 3.9.0

python3.9 -m virtualenv venv39

# Activate virtual environment
venv39\Scripts\activate
```

### 2). Clone MDF repository from Github into the virtual environment
```
git clone https://github.com/ModECI/MDF.git
```

### 3). Change into  the MDF directory
```
cd MDF
```

### 4). Install MDF package
```
pip install .
```

### 5). Install all dependencies
```
pip install .[all]
```

### 6). Change directory into sphinx folder
```
cd docs\sphinx
```

### 7). Create offline documentation in sphinx folder
```
# To allow fresh start when making the documentation
make clean

# To make the documentation
make html
```

### 8). change directory into html folder
```
cd build\html
```

### 9). Run the documentation offline
```
index.html
```


