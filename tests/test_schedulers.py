import pytest


def test_simple_scheduler_main(tmpdir):

    import modeci_mdf.simple_scheduler
    from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW
    import numpy as np

    mdf_formats = ['json','yaml']
    array_formats = [FORMAT_NUMPY, FORMAT_TENSORFLOW]

    # For now, don't make tensorflow a requiremnt...
    try:
        import tensorflow
    except:
        array_formats = [FORMAT_NUMPY]

    for mdf_format in mdf_formats:
        for array_format in array_formats:

            eg = modeci_mdf.simple_scheduler.main("examples/Simple.%s"%mdf_format, array_format=array_format)
            output = eg.enodes['processing_node'].evaluable_outputs['output_1'].curr_value
            assert output==0.6016871801828567

            eg = modeci_mdf.simple_scheduler.main("examples/ABCD.%s"%mdf_format, array_format=array_format)
            output = eg.enodes['D'].evaluable_outputs['output_1'].curr_value
            assert output==0.6298621883736628

            eg = modeci_mdf.simple_scheduler.main("examples/Arrays.%s"%mdf_format, array_format=array_format)
            output = eg.enodes['middle_node'].evaluable_outputs['output_1'].curr_value
            assert output[0,0]==0.5
            assert output[1,1]==4

            eg = modeci_mdf.simple_scheduler.main("examples/States.%s"%mdf_format, array_format=array_format)
            output = eg.enodes['counter_node'].evaluable_outputs['out_port'].curr_value
            assert output==1
            output = eg.enodes['sine_node'].evaluable_outputs['out_port'].curr_value
            assert output==0
