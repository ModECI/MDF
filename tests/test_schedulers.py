import pytest


def test_simple_scheduler_main(tmpdir):

    import modeci_mdf.simple_scheduler

    modeci_mdf.simple_scheduler.main("examples/Simple.json")
    modeci_mdf.simple_scheduler.main("examples/ABCD.yaml")


@pytest.mark.parametrize(
    "fi", ["examples/abc_conditions.json", "examples/abc_conditions.yaml"]
)
def test_condition_scheduler_main(fi):

    import modeci_mdf.condition_scheduler

    modeci_mdf.condition_scheduler.main(fi)
