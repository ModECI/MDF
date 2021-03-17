import pytest
import os


@pytest.mark.xfail(reason='Path issues with defaul JSON model.')
def test_simple_scheduler_main(tmpdir):
    import modeci_mdf.SimpleScheduler
    modeci_mdf.SimpleScheduler.main()