"""
Programmatically defines every ONNX operation as a python callable function. Executing ONNX graphs in this
way somewhat defeats the performance purposes of ONNX since the overhead for each operation will be high. However, this
allows us to test the MDF scheduler (which invokes Python functions) on any MDF model defined over ONNX operations. In
the future, the MDF should probably just compile to ONNX (or some other IR) for execution.
"""

import numpy as np
import onnxruntime as ort
import onnx.defs

# Currently using sklearn2onnx API to define ONNX operations. This dependency can probably be removed pretty easily.
# Do not remove this import even though it appears unused.
import skl2onnx.algebra.onnx_ops

from typing import Dict, Tuple, Any, List, Callable

OpSchema = onnx.defs.OpSchema
FormalParameterOption = OpSchema.FormalParameterOption

# Use the same ONNX opset version that torch is using for defaults now
# from torch.onnx.symbolic_helper import _default_onnx_opset_version as onnx_opset_version
onnx_opset_version = 13

__all__ = [
    "predict_with_onnxruntime",
    "run_onnx_op",
    "get_onnx_ops",
    "get_all_schemas_version",
]


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


def convert_type(v):
    """Helper function to convert types to ONNX compatible types."""

    if type(v) == list:
        v = np.array(v)

    if type(v) == int or type(v) == float:
        v = np.atleast_1d(v)

    if hasattr(v, "dtype") and v.dtype == np.int32:
        v = v.astype(np.int64)

    return v


def run_onnx_op(
    op_name: str,
    inputs: Dict[str, np.array],
    output_names: List[str],
    opset_version: int = onnx_opset_version,
    **attributes,
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

    # If the op name has the onnx namespace prefix, remove it.
    if "onnx::" in op_name:
        op_name = op_name.split("::")[-1]

    inputs = {k: convert_type(v) for k, v in inputs.items()}

    # In the case of the Pad operator, if constant_value argument is passed with different dtype
    # to the data, cast it.
    if op_name == "Pad":
        if "constant_value" in inputs:
            cval = inputs["constant_value"]
            data = list(inputs.values())[0]
            if cval.dtype != data.dtype:
                inputs["constant_value"] = cval.astype(data.dtype)

    op_class = import_class(f"skl2onnx.algebra.onnx_ops.Onnx{op_name}")
    input_names = list(inputs.keys())
    input_vals = list(inputs.values())
    op = op_class(
        *input_names, output_names=output_names, op_version=opset_version, **attributes
    )
    model_def = op.to_onnx(inputs)
    return predict_with_onnxruntime(model_def, *input_vals)


def get_all_schemas_version(max_version: int) -> List[onnx.defs.OpSchema]:
    """
    Enumerate all the OpSchemas available from ONNX.

    Args:
        max_version: Only include up to max_version.

    Returns:
        A list of all OpSchemas
    """
    schemas = {}

    for schema in onnx.defs.get_all_schemas_with_history():
        # Only get ONNX ops available for this version
        if schema.domain == "" and schema.since_version <= max_version:
            if (
                schema.name in schemas
                and schemas[schema.name].since_version < schema.since_version
            ):
                schemas[schema.name] = schema
            elif schema.name not in schemas:
                schemas[schema.name] = schema

    return list(schemas.values())


def get_onnx_ops(opset_version: int = onnx_opset_version) -> List[Dict]:
    """
    Enumerate all available ONNX operations and generate MDF function specifications for each one.

    Args:
        opset_version: The opset version to use.

    Returns:
        A list of MDF function specifications. Each entry is a Dict that is feed directly to
            add_mdf_function.
    """

    mdf_funcspecs = []
    for schema in get_all_schemas_version(opset_version):
        args_list = [input.name for input in schema.inputs]
        params_list = [p for p in schema.attributes]
        args_params_str = ", ".join(args_list + params_list)

        mdf_funcspecs.append(
            dict(
                name=f"onnx::{schema.name}",
                description=schema.doc,
                arguments=args_list,
                expression_string=f"onnx_ops.{schema.name.lower()}({args_params_str})",
            )
        )

    return mdf_funcspecs


def _make_onnx_function(schema: onnx.defs.OpSchema) -> Callable:
    """
    Create and Python callable function from an ONNX OpSchema

    Args:
        schema: The ONNX schema to make a function from.

    Returns:

    """

    def onnx_wrapper(*args, **kwargs):

        # If we are dealing with cosntant, just return it.
        if schema.name == "Constant":
            value = args[0] if len(args) > 0 else list(kwargs.values())[0]
            return value

        # We have a workaround for ConstantOfShape, for some reason shape inference fails
        # if the optional value is passed as a numpy array. Convert to a TensorProto
        elif schema.name == "ConstantOfShape":
            if "value" in kwargs:
                kwargs["value"] = onnx.numpy_helper.from_array(
                    convert_type(kwargs["value"])
                )
            elif len(args) > 1:
                args[1] = onnx.numpy_helper.from_array(convert_type(kwargs["value"]))

        input_names = [inp.name for inp in schema.inputs]

        inputs_dict = {}

        # If the first argument is varidic, then any kwargs that are not in attributes
        # or are not passed as kwargs should passed into the input dict
        if (
            len(schema.inputs) > 0
            and schema.inputs[0].option == FormalParameterOption.Variadic
        ):

            for i, arg in enumerate(args):
                inputs_dict[f"input{i}"] = arg

            remove_list = []
            for kw, arg in kwargs.items():
                if kw not in schema.attributes:
                    inputs_dict[kw] = arg
                    remove_list.append(kw)

            # Remove any kwarg we have assigned to input, it is not an attribute
            for kw in remove_list:
                del kwargs[kw]

        else:
            # First, check if kwargs contains any inputs
            for kw in input_names:
                try:
                    inputs_dict[kw] = kwargs[kw]
                except KeyError:
                    pass

            # Assign any input names that have not yet been assigned by kwargs the remaning positional args
            arg_i = 0
            for inp_name in input_names:
                if inp_name not in inputs_dict:
                    if arg_i < len(args) and arg_i < len(input_names):
                        inputs_dict[inp_name] = args[arg_i]
                        arg_i = arg_i + 1

            # Remove any input argument specified in kwargs
            for input_arg in inputs_dict:
                if input_arg in kwargs:
                    del kwargs[input_arg]

            # If we have any remaining args. Assume they are attributes and assign them in order
            attribute_i = 0
            schema_attributes = list(schema.attributes)
            for arg in [arg for arg in args[arg_i:]]:
                while attribute_i < len(schema.attributes):
                    att_name = schema_attributes[attribute_i]

                    # If this attribute is already in kwargs, skip it
                    if att_name in kwargs:
                        attribute_i = attribute_i + 1
                    else:
                        kwargs[att_name] = arg
                        break

        # Check to make sure all the remaining kwargs are attributes
        for kw in kwargs:
            if kw not in schema.attributes:
                raise ValueError(
                    f"Passed unkown attribute ({kw}) to ONNX op {schema.name}, supported attributes: {list(schema.attributes)}"
                )

        output_names = [out.name for out in schema.outputs]

        out_dict = run_onnx_op(
            op_name=schema.name, inputs=inputs_dict, output_names=output_names, **kwargs
        )

        if len(out_dict) == 1:
            return tuple(out_dict.values())[0]
        else:
            return tuple(out_dict.values())

    return onnx_wrapper


def _define_onnx_functions(opset_version):
    """
    Enumerate all ONNX operators and define Python Callable functions for each one. This kind of defeats the purpose of
    ONNX since we are paying the overhead for invoking each of these functions separately from Python. However, for now,
    this hack will allow us to test any MDF model that is composed of ONNX functions through the scheduler.
    """
    import sys

    current_module = sys.modules[__name__]

    for schema in get_all_schemas_version(opset_version):

        onnx_wrapper = _make_onnx_function(schema)

        # Lets call this function a lowercase version of the opname, follow PEP 8
        func_name = schema.name.lower()

        # Lets add some documentation.
        onnx_wrapper.__doc__ = schema.doc

        setattr(current_module, func_name, onnx_wrapper)


# Define all ONNX operators as functions on this module.
_define_onnx_functions(onnx_opset_version)
