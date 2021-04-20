import pytest


def test_simple_scheduler_main(tmpdir):

    import modeci_mdf.simple_scheduler

    modeci_mdf.simple_scheduler.main("examples/Simple.json")
    modeci_mdf.simple_scheduler.main("examples/ABCD.yaml")
