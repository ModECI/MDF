# 🎇 Contributing Guidelines

This documentation contains a set of guidelines to help new users and potential contributors to MDF.


## 💻 Before Contributing

Welcome to [ModECI/MDF](https://github.com/ModECI/MDF)! Before opening pull requests, make sure that you **read these guidelines**. If you have any doubt on this contributing guide, please feel free to reach out on our [discussion forum](https://github.com/ModECI/MDF/discussions).


## 🙌 Making Contribution

- Install the MDF package and all its dependencies: https://github.com/ModECI/MDF (see below).

- Try running locally the standard MDF examples: https://github.com/ModECI/MDF/tree/main/examples/MDF

- Run the following notebook, altering the network elements along the way to build your own model: https://github.com/ModECI/MDF/blob/main/examples/SimpleExample.ipynb.

- Read the documentation on the elements of the MDF specification: https://mdf.readthedocs.io/en/latest/

- The project uses an issue tracker to keep information about bugs to fix, project features to implement, documentation to write, and more. Potential contributors can look for newcomer-friendly issues by looking for the following issue tags in the [project issue](https://github.com/ModECI/MDF/issues) tracker: good first issue


### 🔖 Steps to Contribute

Following are the steps to guide you:

* Step 1: Fork the repo and Go to your Git terminal and clone it on your machine.

* Step 2: Add a upstream link to main branch in your cloned repo
    ```
    git remote add upstream https://github.com/ModECI/MDF.git
    ```
* Step 3: Keep your cloned repo up to date by pulling from upstream (this will also avoid any merge conflicts while committing new changes)
    ```
    git pull upstream main https://github.com/ModECI/MDF.git
    ```
* Step 4: Create your feature branch (This is a necessary step in order to avoid any disorder in main branch (e.g. bugfix/22))
    ```
    git checkout -b <feature-name>
    ```
* Step 5: Commit all the changes
    ```
    git commit -m "Write a meaningful but small commit message"
    ```
* Step 6: Push the changes for review
    ```
    git push origin <branch-name>
    ```
* Step 7: Create a PR on Github.
     - Don't just hit the create a pull request button, you should write a PR message to clarify why and what are you contributing.
     - Put the hashtag of issue in a commit message for the pull request, and it will show up in the issue itself which will make easy for developers to review your PR based on issue.

### ⚙ MDF Installation Guide

The installation requires Python >= 3.7 (You can install it by the command ```sudo apt-get install python3.8```)

* Step 1: Fork the repo and Go to your Git terminal and clone it on your machine (If done already go directly to next step)
    ```
    git clone https://github.com/ModECI/MDF
    ```
* Step 2 : Install virtualenv(if not installed) for creating virtual environment and installing all the dependencies in it.
    ```
    pip install virtualenv
    ```
* Step 3 : Go to local path of repository
    ```
    cd MDF
    ```
* Step 4 : Create a virtual environment(eg mdf-env)
    ```
    virtualenv mdf-env
    ```
* Step 5 : Activate the virtual environment
    ```
    source mdf-env/bin/activate
    ```
* Step 6 : Install
    ```
    pip install .
    ```

## 📖 Resources

1. Markdown : Markdown is a lightweight markup language like HTML, with plain text formatting syntax.
  * [Markdown Cheat-Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

2. Git : Git is a distributed version-control system for tracking changes in source code during software development. It is designed for coordinating work among programmers, but it can be used to track changes in any set of files.
  * [Videos to get started](https://www.youtube.com/watch?v=xAAmje1H9YM&list=PLeo1K3hjS3usJuxZZUBdjAcilgfQHkRzW)
  * [Cheat Sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)


## 🤔 Need more help?

You can refer to the following articles on basics of Git and Github, in case you are stuck:
- [Forking a Repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)
- [Cloning a Repo](https://help.github.com/en/desktop/contributing-to-projects/creating-an-issue-or-pull-request)
- [How to create a Pull Request](https://opensource.com/article/19/7/create-pull-request-github)
- [Getting started with Git and GitHub](https://towardsdatascience.com/getting-started-with-git-and-github-6fcd0f2d4ac6)
