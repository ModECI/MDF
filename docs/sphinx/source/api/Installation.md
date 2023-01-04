# Installation

## Requirements

Python (>=3.7)

## Quick start

```
pip install modeci_mdf
```

## Installation from source
To install the MDF package from source and run it locally:

### 1) Clone this repository
```
git clone https://github.com/ModECI/MDF.git
```
### 2) Change to the directory
```
cd MDF
```
### 3) Create a virtual environment (e.g. called `mdf-env`)
```
pip install virtualenv
virtualenv mdf-env
```
### 4) Activate the virtual environment
```
source mdf-env/bin/activate
```
### 5) Install the package
```
pip install .

```
Hello world

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
