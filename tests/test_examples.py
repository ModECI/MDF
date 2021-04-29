import pytest
import os
import sys


def test_examples():
    """
    Run the examples and make sure they don't crash.
    """

    # The examples import some modules in the local examples directory. Easiest
    # thing is to just chdir and add current to the path.
    os.chdir("examples")
    sys.path.append(".")

    import simple

    simple.main()
    print("Tested simple model")

    import abcd

    abcd.main()
    print("Tested ABCD model")

    import arrays

    arrays.main()
    print("Tested Arrays model")

    import scaling

    scaling.main()
    print("Tested Scaling model")

    import states

    states.main()
    print("Tested States model")

    # Cleanup, not sure I need this but just to be safe cause this is weird.
    os.chdir("..")
    sys.path.pop()
    print("Done example tests")
