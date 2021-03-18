import pytest
import os
import sys


@pytest.mark.xfail(reason="Model.to_yaml_file() seems to be broken.")
def test_examples():
    """
    Run the examples and make sure they don't crash.
    """

    # The examples import some modules in the local examples directory. Easiest
    # thing is to just chdir and add current to the path.
    os.chdir('examples')
    sys.path.append('.')

    import examples.ABCD
    examples.ABCD.main()

    import examples.Simple
    examples.Simple.main()

    # Cleanup, not sure I need this but just to be safe cause this is weird.
    os.chdir('..')
    sys.path.pop()
