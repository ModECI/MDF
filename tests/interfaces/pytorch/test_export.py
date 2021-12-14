import torch
import torch.nn as nn
import numpy as np
import modeci_mdf.execution_engine
from modeci_mdf.interfaces.pytorch import ABCD_pytorch


from modeci_mdf.utils import load_mdf
from modeci_mdf.interfaces.pytorch import mdf_to_pytorch
from modeci_mdf.execution_engine import EvaluableGraph

from modeci_mdf.utils import load_mdf_json
import json
from pathlib import Path


def test_ABCD():
    base_path = Path(__file__).parent

    filename = "examples/MDF/ABCD.json"
    file_path = (base_path / "../../.." / filename).resolve()
    k = []
    for i in ABCD_pytorch.res:
        print(i.astype(float))
        k.append(round(i.item(0), 3))

    # Get the result of MDF execution
    eg = modeci_mdf.execution_engine.main(str(file_path))
    assert eg.enodes["A"].evaluable_outputs["output_1"].curr_value == k[1]
    assert round(eg.enodes["B"].evaluable_outputs["output_1"].curr_value, 3) == k[2]
    assert round(eg.enodes["C"].evaluable_outputs["output_1"].curr_value, 3) == k[3]
    assert round(eg.enodes["D"].evaluable_outputs["output_1"].curr_value, 3) == k[4]

    # print(np.round(np.array(eg.enodes["A"].evaluable_outputs["output_1"].curr_value)),3)
    # print(ABCD_pytorch.res[1])
    #
    # print(np.array(eg.enodes["B"].evaluable_outputs["output_1"].curr_value))
    # print(ABCD_pytorch.res[2])
    #
    # print(np.array(eg.enodes["C"].evaluable_outputs["output_1"].curr_value))
    # print(ABCD_pytorch.res[3])
    #
    # print(np.round(np.array(eg.enodes["D"].evaluable_outputs["output_1"].curr_value),3))
    # print(np.round(ABCD_pytorch.res[4]),3)


if __name__ == "__main__":
    test_ABCD()
