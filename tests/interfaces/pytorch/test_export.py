import numpy as np
import modeci_mdf.execution_engine
import modeci_mdf.interfaces.pytorch.exporter
from pathlib import Path
from examples.PyTorch.MDF_PyTorch import ABCD_pytorch
from examples.PyTorch.MDF_PyTorch import Arrays_pytorch
from examples.PyTorch.MDF_PyTorch import Simple_pytorch


def test_ABCD():
    base_path = Path(__file__).parent

    filename = "examples/MDF/ABCD.json"
    file_path = (base_path / "../../.." / filename).resolve()
    k = []
    for i in ABCD_pytorch.res:
        k.append(round(i.item(0), 3))

    # Get the result of MDF execution
    eg = modeci_mdf.execution_engine.main(str(file_path))
    assert eg.enodes["A"].evaluable_outputs["output_1"].curr_value == k[1]
    assert round(eg.enodes["B"].evaluable_outputs["output_1"].curr_value, 3) == k[2]
    assert round(eg.enodes["C"].evaluable_outputs["output_1"].curr_value, 3) == k[3]
    assert round(eg.enodes["D"].evaluable_outputs["output_1"].curr_value, 3) == k[4]


def test_Arrays():
    base_path = Path(__file__).parent

    filename = "examples/MDF/Arrays.json"
    file_path = (base_path / "../../.." / filename).resolve()

    k = Arrays_pytorch.res[1]

    # # Get the result of MDF execution
    eg = modeci_mdf.execution_engine.main(str(file_path))
    output = eg.enodes["middle_node"].evaluable_outputs["output_1"].curr_value
    assert output[0, 0] == k[0, 0]
    assert output[0, 1] == k[0, 1]
    assert output[1, 1] == k[1, 1]


def test_Simple():
    base_path = Path(__file__).parent

    filename = "examples/MDF/Simple.json"
    file_path = (base_path / "../../.." / filename).resolve()
    k = []
    for i in Simple_pytorch.res:
        k.append(round(i.item(0), 3))

    # Get the result of MDF execution
    eg = modeci_mdf.execution_engine.main(str(file_path))
    assert eg.enodes["input_node"].evaluable_outputs["out_port"].curr_value == k[0]
    assert (
        round(eg.enodes["processing_node"].evaluable_outputs["output_1"].curr_value, 3)
        == k[1]
    )


if __name__ == "__main__":
    test_ABCD()
    test_Arrays()
    test_Simple()
