"""
Test some individual ONNX operator calls. We should probably find a better way to
test these more exhaustively by hooking into ONNX's testing framework. For now,
I have hand coded some tests for operators that are used in the examples we have
worked on. There could be broken operators.
"""
import numpy as np
import modeci_mdf.functions.onnx as onnx_ops


def test_conv():
    """Test ONNX Conv function"""
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
    """Test ONNX Pad function"""
    x = np.zeros((3, 2))
    value = np.array(1.5)
    pads = np.array([0, 1, 0, 1]).astype(np.int64)

    out = onnx_ops.pad(x, pads, value, mode="constant")

    # Try attributes without keyword
    out2 = onnx_ops.pad(x, pads, value, "constant")

    assert np.all(out == out2)


def test_pad_diff_types():
    """Check if Pad can handle the case were a different type is passed to constant_value than the type of the data"""

    args = {
        "data": np.zeros((1, 48, 32, 32), dtype=np.float32),
        "pads": np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64),
        "constant_value": 1.5,
        "mode": "constant",
    }

    onnx_ops.pad(**args)


def test_unsqueeze():
    """Test ONNX unsqueeze function."""
    data = np.zeros((3, 2))
    axes = [0]

    out = onnx_ops.unsqueeze(data=data, axes=axes)

    assert out.ndim == 3
    assert out.shape == (1, 3, 2)


def test_mul():
    """Test element-wise tensor multiplication (Mul)"""
    A = np.ones((1, 3)) * 2.0
    B = np.ones((3, 1))
    assert np.allclose(A * B, onnx_ops.mul(A, B))

    A = 1
    B = 2
    assert np.allclose(A * B, onnx_ops.mul(A, B))


def test_constantofshape():
    """Test ConstantOfShape function."""
    out = onnx_ops.constantofshape(np.array([4, 4], dtype=np.int64), value=[0])
    assert np.allclose(out, np.zeros((4, 4), dtype=np.int64))


def test_concat():
    """Test ONNX Concat function. This is a variable number of inputs operator."""
    input = (np.ones(3), np.ones(3), np.ones(3))
    out = onnx_ops.concat(*input, axis=0)
    assert np.allclose(out, np.concatenate(input, axis=0))


def test_maxpool():
    """Test ONNX Concat function. This is a variable number of inputs operator."""
    out = onnx_ops.maxpool(
        np.ones((1, 3, 32, 32)).astype(np.float32), kernel_shape=[2, 2]
    )
    assert True
