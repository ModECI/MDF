from pathlib import Path
import os
import sys
from modeci_mdf.utils import load_mdf
from modeci_mdf.interfaces.pytorch.exporter import mdf_to_pytorch


def main(filename):
    base_path = Path(__file__).parent
    file_path = str((base_path / "../../.." / filename).resolve())

    print("Converting MDF model in %s to PyTorch" % file_path)

    model_input = file_path.replace(os.sep, "/")

    mdf_model = load_mdf(model_input)

    if "Translated" in model_input:
        pytorch_model = mdf_to_pytorch(
            mdf_model,
            model_input,
            eval_models=False,
            version="mdf.0",  #  (MDF "zero" - a simplified form of MDF)
        )
    else:
        pytorch_model = mdf_to_pytorch(
            mdf_model,
            model_input,
            eval_models=False,
            version="mdf.s",  #  (MDF "stateful" - full MDF allowing stateful parameters)
        )


if __name__ == "__main__":

    sample_examples = [
        "examples/MDF/Simple.json",
        "examples/MDF/ABCD.json",
        "examples/MDF/Arrays.json",
        "examples/MDF/translation/Translated_Arrays.json",
        "examples/MDF/translation/Translated_Simple.json",
        "examples/MDF/translation/Translated_ABCD.json",
    ]

    if "-all" in sys.argv:
        for ex in sample_examples:
            main(ex)

    elif "-test" in sys.argv:
        for ex in sample_examples:
            if not "Translated" in ex:
                main(ex)
    else:
        filename = "examples/MDF/Simple.json"
        main(filename)
