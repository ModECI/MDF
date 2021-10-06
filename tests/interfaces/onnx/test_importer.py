from modeci_mdf.utils import load_mdf
from modeci_mdf.interfaces.onnx import mdf_to_onnx

import onnx
from pathlib import Path

from modeci_mdf.execution_engine import EvaluableGraph
import onnxruntime as rt
import onnxruntime.backend as backend
import numpy as np


def test_ab():
    base_path = Path(__file__).parent

    filename = "examples/ONNX/ab.json"
    file_path = (base_path / "../../.." / filename).resolve()

    # Load the MDF model
    mdf_model = load_mdf(str(file_path))

    # Test input
    test_input = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

    # Get the result of MDF execution
    mdf_executable = EvaluableGraph(mdf_model.graphs[0],
                                    verbose=False)
    # TODO: the int type cast is necessaryf or now because the nodes' parameters are constants and inputs must have
    #  the same type
    mdf_executable.evaluate(initializer={"input": test_input.astype(int)})
    mdf_output = mdf_executable.enodes['Mul_3'].evaluable_outputs['_4'].curr_value

    # Get the translated ONNX model
    onnx_models = mdf_to_onnx(mdf_model)

    # Bluffing onnx that our model is 13 when it is actually 15. This is needed for older onnxruntime
    # installations to run this model. See https://github.com/onnx/onnx/issues/3205
    onnx_models[0].opset_import[0].version = 13

    # Get the result of running the ONNX model
    session = backend.prepare(onnx_models[0])
    onnx_output = session.run(test_input)  # run returns a list with the actual result and type
    onnx_res_output = np.array(onnx_output[0])
    # print(f"Output calculated by onnxruntime: {onnx_res_output} and MDF: {mdf_output.astype(float)}")

    assert np.array_equal(onnx_res_output, mdf_output)


def test_abc():
    base_path = Path(__file__).parent

    filename = "examples/ONNX/abc_basic-mdf.json"
    file_path = (base_path / "../../.." / filename).resolve()

    # Load the MDF model
    mdf_model = load_mdf(str(file_path))

    # Test input
    test_input = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

    # Get the result of MDF execution
    mdf_executable = EvaluableGraph(mdf_model.graphs[0],
                                    verbose=False)
    mdf_executable.evaluate(initializer={"input": test_input})
    mdf_output = mdf_executable.enodes['Cos_2'].evaluable_outputs['_3'].curr_value

    # Get the translated ONNX model
    onnx_models = mdf_to_onnx(mdf_model)

    # Bluffing onnx that our model is 13 when it is actually 15. This is needed for older onnxruntime
    # installations to run this model. See https://github.com/onnx/onnx/issues/3205
    onnx_models[0].opset_import[0].version = 13
    
    # Get the result of running the ONNX model
    session = backend.prepare(onnx_models[0])
    onnx_output = session.run(test_input)  # run returns a list with the actual result and type
    onnx_res_output = np.array(onnx_output[0])

    assert np.array_equal(onnx_res_output, mdf_output)
    #reference_onnx_filename = "examples/ONNX/abc_basic.onnx"
    #reference_file_path = (base_path / "../../.." / reference_onnx_filename).resolve()
    #reference_onnx_model = onnx.load(reference_file_path)

    #for onnx_model in onnx_models:
    #    assert onnx.checker.check_model(onnx_model) is None


if __name__ == '__main__':
    test_ab()
    test_abc()
