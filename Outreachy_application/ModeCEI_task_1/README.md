# Installation and Documentation Guidelines for ModECI Libraries.

## Python Recommendation
Python versions between **Py v3.7** to **Py v3.9** are best suited and the most compatible for the ModECI libraries, we'd see why in subsequentlines. I downloaded python using the link [here](https://www.python.org/downloads/)
After installating the recommended python versions to my machine, I add the different form **py v3.7 to py v3.11** python versions to PATH by holding the **windows + R**. This was a very insightful experiment because through this I was able to understand ModeCEI and it's dependencies with different python versions.

This launched the windows run windows dialog box. In this dialog box, I will pasted the text below and hit enter:
```
sysdm.cpl
```

Immediately after this the system properties prompt popped up. I selected **advanced** then selected **environment variable** from the advanced tab. Clicking on this opened up the environment variable console where I added the python PATH to both the user and system variables.

## Cloning the Local Repository form Source
For the next step, I clone the MDF remote repository Into my local machine using the command below:

```
git clone https://github.com/ModECI/MDF.git
```

## Setting up The Environment
After downloading all python versions from v3.7 to v3.11, I then proceeed to setting up our environment by installing all necessary ModECI libraries' dependencies. Before doing this, I created a parent directory that'd house the **virtual environment(s) files, MDF library and other similar file of importance**.

## Creating our Virtual Environment
As a rule of thumb, I created a virtual environment for all the recommended python versions and installed all their dependencies. I did this to ensure there were no possibilities of comflict with other unsupported python versions that'd be installed by me in the futre.
I create a virtual environment by first changing directory into the the parent directory and ran the code below to create the virtual environmemt:

```
python<version> -m venv <environment name>
```

As soon as I had created our virtual environment, I activate my vitual environment(s) by running the command:

```
<environment name>\Scripts\Activate.bat
```

After activativating my virtual environment, I changed directory into the MDF local repository then ran the command:

```
pip install .[all]
```

## Observation 1
While running the file MDF/examples/SimpleExample.ipynb the code block **Generate a graph image from the model**, kept failing because of a dependency issue with the package Graphviz.

## Solving the Graphviz Dependency Issues
I solved this issue by installing the graphviz dependency, I did this by running the code below:

```
pip install graphviz
```

Along side installing the graphviz dependancy with pip, I also need to installed the graphviz application package from [here](https://graphviz.org/download/). Similar to how I added python to PATH, I added the graphviz directory containing the dot.exe extension to the system PATH.

# Generating Documentation on MDF

## Observation 2

* Sphinx is compatible with Python v3.10 although it best to use Python v3.9 because it runs the make command 
without any issues.

## Solving Issues with Runnig Sphinx

Generating an MDF documentation can be done using the package called Sphinx. Sphinx is a documentation generator or a tool that translates a set of plain text source files into various output formats, automatically producing cross-references, indices, etc. 
To run generate a document, the make command is a requirement. This is a command native to the GNU Linux system.

### Step 1: Install Chocolatey
To install the make command, I the followed steps:
* I install chocolatey
  * I opened powershell as administrator
  * I ran 
  ```
  Get-ExecutionPolicy
  ```
  * If it returns **Restricted**
  * I ran 
  ```
  Set-ExecutionPolicy AllSigned` or `Set-ExecutionPolicy Bypass -Scope Process
  ```
  * I ran 
  ```
  Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  ```
  * I waited for a few moment for it to install
  * I finally I `run choco -V` to comfirm installantion
  
### Step 2: Install make
I installed the make command,by running running the command:

```
choco install make
```

### Step 3: Create a Python v3.9 Virtual Environment
I observed Python v3.9 was the most stable for running the make command, thus creating 3.9 specific virtual environment was most prefered by the dependencies and commands. I created a virtual environment by running the command:

```
python<version> -m venv <environment name>
```

### Step 4: Activate the Created Environment
After creating the virtual environment I navigated to the specific parent directory where the virtual environment was created and ran the command below:

```
<environment name>\Scripts\Activate.bat
```

### Cloning  the MDF Repository
As soon as the virtual environment had been created I went to [MDF](https://github.com/ModECI/MDF) where i cloned the repository using:

```
git clone https://github.com/ModECI/MDF.git
```

### Setting up Sphinx and Other dependencies
When the virtual environment was all set and the MDF had been cloned, I install all ModECI dependencies using:

```
pip install .[all]
```

In most cases, sphinx is not usually installed when we run **"pip install .[all]"**, its recommended to run the following command after running the **pip install .[all]**:

```
pip install sphinx== 3.5.4
```

### Generating an MDF Documentation
I generated a documentation, by running the command:
This resets previously generated or loaded documentation files.

```
make clean
```

Finally I ran the command:
This command generated a fresh file and directory where the file was located.

```
make html
```

### Viewing the Documentation Offline
Navigate into the newest file in the sphinx directory, this is usually a file named **build**. Enter into the build file and change directory into the file named html and click one the html file. This will load a browers instance and we can view the documentation created.

### Conclusion
* Python 3.11 is not compatible with the ModECI library and its dependencies.
* Python 3.10 is stated to be compatible with ModECEI library according to the documentation, despite this, I encounted some minor instability issues while generating and working out a workflow for generating documentations with the ModECI library. The instable issues was observed while the running the make command.
* Python 3.9 seems to be the most stable version according to this exercised.
* **pip install .[all]** did not install some dependencies such as graphviz and sphinx. This had to be done separately.
All this was made possible as result of testing with the different python libraries.