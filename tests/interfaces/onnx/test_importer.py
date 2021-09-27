from modeci_mdf.utils import load_mdf
from modeci_mdf.interfaces.onnx import mdf_to_onnx

import onnx
import unittest
from pathlib import Path

class SimpleTestCase(unittest.TestCase):
    def test_ab(self):
        base_path = Path(__file__).parent

        filename = "examples/ONNX/ab.json"
        file_path = (base_path / "../../.." / filename).resolve()
        mdf_model = load_mdf(str(file_path))
        onnx_models = mdf_to_onnx(mdf_model)

        reference_onnx_filename = "examples/ONNX/ab.onnx"
        reference_file_path = (base_path / "../../.." / reference_onnx_filename).resolve()
        reference_onnx_model = onnx.load(reference_file_path)

        for onnx_model in onnx_models:
            assert onnx.checker.check_model(onnx_model) is None

    def test_abc(self):
        base_path = Path(__file__).parent

        filename = "examples/ONNX/abc_basic-mdf.json"
        file_path = (base_path / "../../.." / filename).resolve()
        mdf_model = load_mdf(str(file_path))
        onnx_models = mdf_to_onnx(mdf_model)

        reference_onnx_filename = "examples/ONNX/abc_basic.onnx"
        reference_file_path = (base_path / "../../.." / reference_onnx_filename).resolve()
        reference_onnx_model = onnx.load(reference_file_path)

        for onnx_model in onnx_models:
            assert onnx.checker.check_model(onnx_model) is None


if __name__ == '__main__':
    unittest.main()
