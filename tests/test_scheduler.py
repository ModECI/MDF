import pytest


def test_execution_engine_main(tmpdir):

    import modeci_mdf.execution_engine
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

            eg = modeci_mdf.execution_engine.main("examples/MDF/Simple.%s"%mdf_format, array_format=array_format)
            output = eg.enodes['processing_node'].evaluable_outputs['output_1'].curr_value
            assert output==0.6016871801828567

            eg = modeci_mdf.execution_engine.main("examples/MDF/ABCD.%s"%mdf_format, array_format=array_format)
            output = eg.enodes['D'].evaluable_outputs['output_1'].curr_value
            assert (-1.7737500239216304-output<1e-9)

            eg = modeci_mdf.execution_engine.main("examples/MDF/Arrays.%s"%mdf_format, array_format=array_format)
            output = eg.enodes['middle_node'].evaluable_outputs['output_1'].curr_value
            assert output[0,0]==0.5
            assert output[1,1]==4

            eg = modeci_mdf.execution_engine.main("examples/MDF/States.%s"%mdf_format, array_format=array_format)
            output = eg.enodes['counter_node'].evaluable_outputs['out_port'].curr_value
            assert output==1
            output = eg.enodes['sine_node'].evaluable_outputs['out_port'].curr_value
            assert output==0


def test_execution_engine_onnx(tmpdir):

    import modeci_mdf.execution_engine
    import numpy as np

    mdf_formats = ['json','yaml']
    from neuromllite.utils import FORMAT_NUMPY

    array_format = FORMAT_NUMPY

    for mdf_format in mdf_formats:

            eg = modeci_mdf.execution_engine.main("examples/ONNX/ab.%s"%mdf_format, array_format=array_format)
            output = eg.enodes['Mul_3'].evaluable_outputs['_4'].curr_value
            assert output==5



_abc_conditions_expected_output = [
    {"input0"},
    {"A"},
    {"A"},
    {"B"},
    {"A"},
    {"C"},
    {"A"},
    {"B"},
    {"A"},
    {"A"},
    {"C", "B"},
    {"A"},
]

@pytest.mark.parametrize(
    "fi, expected_output",
    [
        ("examples/MDF/abc_conditions.json", _abc_conditions_expected_output),
        ("examples/MDF/abc_conditions.yaml", _abc_conditions_expected_output),
    ],
)
def test_condition_scheduler_main(fi, expected_output):

    import modeci_mdf.execution_engine

    eg = modeci_mdf.execution_engine.main(fi)
    output = [set([n.id for n in nodes]) for nodes in eg.scheduler.execution_list[None]]

    assert output == expected_output
