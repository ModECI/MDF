# Contribution Guidelines

This documentation contains a set of guidelines to help new users and potential contributors to MDF.


## Before Contributing

Before opening pull requests, make sure that you **read these guidelines**. If you have any doubt on this contributing guide, please feel free to reach out on our [discussion forum](https://github.com/ModECI/MDF/discussions).


## Making Contributions

- Install the MDF package and all its dependencies: https://github.com/ModECI/MDF (see [here](https://mdf.readthedocs.io/en/latest/api/Installation.html) for full details).

- Try running locally the standard MDF examples: https://github.com/ModECI/MDF/tree/main/examples/MDF

- Run the following notebook, altering the network elements along the way to build your own model: https://github.com/ModECI/MDF/blob/main/examples/SimpleExample.ipynb.

- Read the documentation on the elements of the MDF specification: https://mdf.readthedocs.io

- The project uses an issue tracker to keep information about bugs to fix, project features to implement, documentation to write, and more. Potential contributors can look for newcomer-friendly issues by looking for the following issue tags in the [project issue](https://github.com/ModECI/MDF/issues) tracker: `good first issue`.


### Steps to Contribute

Following are the steps to guide you to making your own fork of the MDF repository, making changes and submitting them as contributions:

#### Step 1
Create and activate a **virtual environment for MDF** as outlined in the main [installation guide](https://mdf.readthedocs.io/en/latest/api/Installation.html).

#### Step 2

[Fork the MDF repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) on the GitHub website, and then go to your terminal and clone it on your machine.

```
git clone https://github.com/<your_fork_name>/MDF
```

#### Step 3

Add an upstream link to main branch in your cloned repo.

```
git remote add upstream https://github.com/ModECI/MDF.git
```

#### Step 4

Keep your cloned repo up to date by pulling from upstream (this will also avoid any merge conflicts while committing new changes)

```
git pull upstream main https://github.com/ModECI/MDF.git
```

#### Step 5

Create your feature branch. Note: it is useful to give this a name relevant to the issue being addressed e.g. `feat/my_new_feature` or `bugfix/123` (to fix issue #123)

```
git checkout -b <feature-name>
```

#### Step 6
**Make your changes!** Run the tests in [test_all.sh](https://github.com/ModECI/MDF/blob/main/test_all.sh) to make sure all tests are passing locally.

#### Step 7
Format your code. We use a standard format ([black](https://github.com/psf/black)) for all our code, as this minimises the changes between commits especially when people have different coding styles. Install [pre-commit](https://pre-commit.com/) using `pip install pre-commit` and type `pre-commit run --all-files` at the top level MDF directory to format the code. This will change all the relevant files to the correct formatting before you commit. This formatting is checked by our [GitHub Actions tests](https://github.com/ModECI/MDF/actions) and will fail if the code is not correctly formatted.

#### Step 8
Commit your the changes. Note: if you have run the [test_all.sh](https://github.com/ModECI/MDF/blob/main/test_all.sh), many of the image files will have been regenerated (and may show as changed even though they are identical). Don't commit these unless you know there is an actual change.

```
git commit -m "A meaningful, concise commit message"
```

#### Step 9
Push the changes to your fork

```
git push origin <branch-name>
```

#### Step 10
Create a PR on GitHub.

- Don't just hit the create a pull request button, you should write a detailed PR message to clarify why and what are you contributing.
- Put the hashtag of a relevant issue in a commit message for the pull request (e.g. #123), and it will show up in the issue itself which will make easy for developers to review your PR based on the issue.


## Resources

1. Markdown : Markdown is a lightweight markup language like HTML, with plain text formatting syntax.
  * [Markdown Cheat-Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

2. Git : Git is a distributed version-control system for tracking changes in source code during software development. It is designed for coordinating work among programmers, but it can be used to track changes in any set of files.
  * [Videos to get started](https://www.youtube.com/watch?v=xAAmje1H9YM&list=PLeo1K3hjS3usJuxZZUBdjAcilgfQHkRzW)
  * [Cheat Sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)


## Need more help?

You can refer to the following articles on basics of Git and Github, in case you are stuck:
- [Forking a Repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)
- [Cloning a Repo](https://help.github.com/en/desktop/contributing-to-projects/creating-an-issue-or-pull-request)
- [How to create a Pull Request](https://opensource.com/article/19/7/create-pull-request-github)
- [Getting started with Git and GitHub](https://towardsdatascience.com/getting-started-with-git-and-github-6fcd0f2d4ac6)
