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

example_exclusion_strings = [".reconstructed.py", "generate_json_and_scripts.py"]


class TestExamples:
    example_scripts = []

    # Core MDF tests
    @pytest.mark.coremdf
    def test_core(self):
        example_core_scripts = (
            glob.glob("examples/MDF/**/*.py")
            + glob.glob("examples/ACT-R/*.py")
            + glob.glob("examples/ONNX/*.py")
            + glob.glob("examples/PyTorch/*.py")
            + glob.glob("examples/WebGME/*.py")
        )
        self.example_scripts = [
            pytest.param(script) if (Path(script) in example_core_scripts) else script
            for script in example_core_scripts
            if all(e not in script for e in example_exclusion_strings)
        ]
        print(self.example_scripts)

    # NeuroML tests
    @pytest.mark.neuromltest
    def test_neuroml(self):
        example_neuroml_scripts = glob.glob("examples/NeuroML/*.py")
        self.example_scripts = [
            pytest.param(script)
            if (Path(script) in example_neuroml_scripts)
            else script
            for script in example_neuroml_scripts
            if all(e not in script for e in example_exclusion_strings)
        ]
        print(self.example_scripts)

    # PsyNeuLink tests
    @pytest.mark.psyneulinktest
    @pytest.mark.skipif(
        "psyneulink" not in sys.modules, reason="requires the psyneulink library"
    )
    def test_pnl(self):
        import psyneulink

        example_pnl_scripts = glob.glob("examples/PsyNeuLink/*.py")
        self.example_scripts = [
            pytest.param(script)
            if (
                Path(script) in example_pnl_scripts
                and not hasattr(psyneulink, "TimeInterval")
            )
            else script
            for script in example_pnl_scripts
            if all(e not in script for e in example_exclusion_strings)
        ]
        print(self.example_scripts)

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


@pytest.mark.parametrize("script", TestExamples.example_scripts)
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
