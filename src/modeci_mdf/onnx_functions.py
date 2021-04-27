import numpy as np
import onnxruntime as ort
from skl2onnx.algebra.onnx_ops import OnnxPad, OnnxConv  # noqa

from typing import Dict, Tuple, Any, List


def import_class(name: str) -> Any:
    """Import from a module specified by a string"""
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def predict_with_onnxruntime(model_def, *inputs) -> Dict[str, np.array]:
    """
    Simple helper to run an ONNX model with a set of inputs.

    Args:
        model_def: The ONNX model to run.
        *inputs: Input values to pass to the model.

    Returns:
        A dict of output values, keys are output names for the model. Values are
        the output values of the model.
    """
    sess = ort.InferenceSession(model_def.SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    dinputs = {name: input for name, input in zip(names, inputs)}
    res = sess.run(None, dinputs)
    names = [o.name for o in sess.get_outputs()]
    return {name: output for name, output in zip(names, res)}


def run_onnx_op(
    op_name: str, inputs: Dict[str, np.array], output_names: List[str], **attributes
):
    """
    Simple helper function that invokes a single ONNX operator with
    inputs and attibutes and returns the results. This isn't typically done
    in ONNX because graphs usually consist of more than one operation.
    This wrapper probably creates a significant amount of overhead for
    but if we want to execute and ONNX graph op by op it is the easiest
    thing to do.

    Args:
        op_name: The name of the operation to run, (Conv, Pad, etc.)
        inputs: A dict keyed by input name where the values are the input values to pass to the operation.
        output_names: The names to use for the output values.
        **attributes: Any additional attributes for the ONNX operation.

    Returns:
        A dict of output values, keys are output_names. Values are
        the output values of the operation.
    """
    op_class = import_class(f"skl2onnx.algebra.onnx_ops.Onnx{op_name}")
    input_names = list(inputs.keys())
    input_vals = list(inputs.values())
    op = op_class(*input_names, output_names=output_names, **attributes)
    model_def = op.to_onnx(inputs)
    return predict_with_onnxruntime(model_def, *input_vals)
