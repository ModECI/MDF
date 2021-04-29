"""
Test some individual ONNX operator calls.
"""
import numpy as np
from modeci_mdf.onnx_functions import run_onnx_op

import modeci_mdf.onnx_functions as onnx_ops


def test_conv():
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]
            ]
        ]
    ).astype(np.float32)
    W = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ]
    ).astype(np.float32)

    out = onnx_ops.conv(x, W)


def test_pad():
    x = np.zeros((3, 2))
    value = np.array(1.5)
    pads = np.array([0, 1, 0, 1]).astype(np.int64)

    out = onnx_ops.pad(x, pads, value, mode="constant")

    # Try attributes without keyword
    out2 = onnx_ops.pad(x, pads, value, "constant")

    assert np.all(out == out2)


def test_unsqueeze():
    data = np.zeros((3, 2))
    axes = [0]

    out = onnx_ops.unsqueeze(data=data, axes=axes)

    assert out.ndim == 3
    assert out.shape == (1, 3, 2)
