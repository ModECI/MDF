import pytest


def test_simple_scheduler_main(tmpdir):

    import modeci_mdf.simple_scheduler

    modeci_mdf.simple_scheduler.main("examples/Simple.json")
    modeci_mdf.simple_scheduler.main("examples/Simple.yaml")

    modeci_mdf.simple_scheduler.main("examples/ABCD.json")
    modeci_mdf.simple_scheduler.main("examples/ABCD.yaml")

    modeci_mdf.simple_scheduler.main("examples/Arrays.json")
    modeci_mdf.simple_scheduler.main("examples/Arrays.yaml")

    modeci_mdf.simple_scheduler.main("examples/States.json")
    modeci_mdf.simple_scheduler.main("examples/States.yaml")
