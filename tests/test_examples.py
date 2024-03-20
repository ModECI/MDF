"""
Runs and tests everything in the examples folder.
"""

import pytest
import glob
import runpy
import os
import sys
import copy

from distutils.dir_util import copy_tree
from pathlib import Path

# Set this to True if you want for example scripts to be run from their current location in the repo rather than
# copied to a pytest temp directory. WARNING: setting this to True will cause files that these scripts generate to be
# re-generated.
run_example_scripts_in_repo = True

example_scripts = glob.glob("examples/**/*.py", recursive=True)
example_mdf_scripts = {
    Path(f) for f in glob.glob("examples/MDF/**/*.py", recursive=True)
}
example_pnl_scripts = {
    Path(f) for f in glob.glob("examples/PsyNeuLink/**/*.py", recursive=True)
}
example_exclusion_strings = [
    ".reconstructed.py",
    "generate_json_and_scripts.py",
    "pytorch_ddm.py",
]

# Filter any excluded example scripts
example_scripts = [
    script
    for script in example_scripts
    if all(e not in script for e in example_exclusion_strings)
]


@pytest.fixture(scope="session")
def example_tmp_dir(tmpdir_factory):
    """
    Create a temporary directory to run all example scripts from.
    This is a session scoped fixture so it is shared for all tests.
    """
    tmpdir = tmpdir_factory.mktemp("examples", numbered=False)
    copy_tree("examples", tmpdir.strpath)
    return tmpdir


@pytest.fixture(autouse=True)
def chdir_back_to_root(mocker):
    """
    This fixture sets up and tears down state before each example is run. Certain examples
    require that they are run from the local directory in which they reside. This changes
    directory and adds the local directory to sys.path. It reverses this after the test
    finishes.
    """

    # Get the current directory before running the test
    cwd = os.getcwd()

    # Some of the scripts do plots. Lets patch matplotlib plot so tests don't hang
    mocker.patch("matplotlib.pyplot.show")
    mocker.patch("matplotlib.pyplot.figure")

    # Cache the path so we can reset it after the test, the examples/MDF tests require
    # setting the path
    old_sys_path = copy.copy(sys.path)

    yield

    sys.path = old_sys_path

    # We need chdir back to root of the repo
    os.chdir(cwd)


@pytest.mark.parametrize("script", example_scripts)
@pytest.mark.parametrize("additional_args", [["-run"]])
def test_example(script, example_tmp_dir, additional_args):
    """
    Run the examples/MDF
    """

    # Change directory to the the parent of the examples temp directory
    # so that we can find the absolute path to the script we are testing
    # within the temporary directory and not the root of the repo.
    os.chdir(str(Path(example_tmp_dir).parent))

    # Get the full path for the script
    full_script_path = os.path.abspath(script)

    # Some of the scripts in examples/MDF import from the local directory. So lets run from the scripts
    # local directory.
    dir_path = os.path.dirname(os.path.realpath(full_script_path))
    os.chdir(dir_path)

    # If this is one of the example/MDF scripts, we need to append example/MDF to sys.path
    if Path(script) in example_mdf_scripts:
        sys.path.append(dir_path)

    print(f"Running script {full_script_path} in working dir {os.getcwd()}")
    orig_argv = sys.argv
    sys.argv = [os.path.basename(full_script_path)] + additional_args
    runpy.run_path(os.path.basename(full_script_path), run_name="__main__")
    sys.argv = orig_argv
