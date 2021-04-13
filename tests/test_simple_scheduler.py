import pytest

@pytest.mark.xfail(reason='Path issues with default JSON model.')
def test_simple_scheduler_main(tmpdir):
    import modeci_mdf.simple_scheduler
    modeci_mdf.simple_scheduler.main()
