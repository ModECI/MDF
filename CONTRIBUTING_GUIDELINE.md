# 🎇Contributing Guidelines

This documentation contains a set of guidelines to help you during the contribution process. 


## 💻Before Contributing

Welcome to [ModECI/MDF](https://github.com/ModECI/MDF). Before sending your pull requests, make sure that you **read the whole guidelines**. If you have any doubt on the contributing guide, please feel free to reach out [here](https://github.com/ModECI/MDF/discussions/213)


## 🙌Project Contribution

- Install the MDF package and all dependencies: https://github.com/ModECI/MDF.

- Run locally the standard MDF examples: https://github.com/ModECI/MDF/tree/main/examples/MDF

- Run the following notebook, altering the network elements along the way to build your own model: https://github.com/ModECI/MDF/blob/main/examples/SimpleExample.ipynb.

- Read the documentation on the elements of the MDF specification: https://mdf.readthedocs.io/en/latest/

- The project uses an issue tracker to keep information about bugs to fix, project features to implement, documentation to write, and more. Applicants can look for newcomer-friendly issues to use for their first contributions by looking for the following issue tags in the [project issue](https://github.com/ModECI/MDF/issues) tracker: good first issue

- The applicant should look at the documentation for MDF at https://mdf.readthedocs.io, the repository for MDF at https://github.com/ModECI/MDF, and the existing notebook example showing how to build an MDF model at: https://github.com/ModECI/MDF/blob/main/examples/SimpleExample.ipynb.


### 🔖Steps to Contribute

Following are the steps to guide you:

* Step 1: Fork the repo and Go to your Git terminal and  clone it on your machine.
* Step 2: Add a upstream link to main branch in your cloned repo
    ```
    git remote add upstream https://github.com/ModECI/MDF.git
    ```
* Step 3: Keep your cloned repo upto date by pulling from upstream (this will also avoid any merge conflicts while committing new changes)
    ```
    git pull upstream main https://github.com/ModECI/MDF.git
    ```
* Step 4: Create your feature branch (This is a necessary step in order to avoid any disorder in main branch(ie:bugfix22/update-xfile))
    ```
    git checkout -b <feature-name>
    ```
* Step 5: Commit all the changes (Write commit message as "Small Message")
    ```
    git commit -m "Write a meaningfull but small commit message"
    ```
* Step 6: Push the changes for review
    ```
    git push origin <branch-name>
    ```
* Step 7: Create a PR on Github. 
     - Don't just hit the create a pull request button, you should write a PR message to clarify why and what are you contributing.
     - Put the hashtag of issue in a commit message for the pull request, and it will show up in the issue itself which will make easy for mentors to review your PR based on issue.

### ⚙MDF Installation Guide

* Step 1: Fork the repo and Go to your Git terminal and clone it on your machine.(If done already go directly to next step)
    ```
    git clone https://github.com/ModECI/MDF
    ```
* Step 2 : Go to local path of repo
    ```
    cd local/path/to/MDF
    ```
* Step 3 : Install
    ```
    pip install .
    ```
   

### 🔨Note:

> - Kindly do not edit/delete someone else's code in this repository. You can insert new files/folder in this repository.

> - Give a meaningful name to whatever file or folder you are adding. 

## 📖Resources

1. Markdown : Markdown is a lightweight markup language like HTML, with plain text formatting syntax. 
  * [Markdown Cheat-Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

2. Git : Git is a distributed version-control system for tracking changes in source code during software development. It is designed for coordinating work among programmers, but it can be used to track changes in any set of files.
  * [Videos to get started](https://www.youtube.com/watch?v=xAAmje1H9YM&list=PLeo1K3hjS3usJuxZZUBdjAcilgfQHkRzW)
  * [Cheat Sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)


## 🤔Need more help?

You can refer to the following articles on basics of Git and Github, in case you are stuck:
- [Forking a Repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)
- [Cloning a Repo](https://help.github.com/en/desktop/contributing-to-projects/creating-an-issue-or-pull-request)
- [How to create a Pull Request](https://opensource.com/article/19/7/create-pull-request-github)
- [Getting started with Git and GitHub](https://towardsdatascience.com/getting-started-with-git-and-github-6fcd0f2d4ac6)

