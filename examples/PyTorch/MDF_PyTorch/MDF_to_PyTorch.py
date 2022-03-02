from pathlib import Path
import os
from modeci_mdf.utils import load_mdf
from modeci_mdf.interfaces.pytorch.exporter import mdf_to_pytorch

# sample_examples=["examples/MDF/translation/Translated_Arrays.json", "examples/MDF/translation/Translated_Simple.json",
#                "examples/MDF/translation/Translated_ABCD.json", "examples/MDF/ABCD.json", "examples/MDF/Arrays.json"
#                  "examples/MDF/Simple.json"]
def main(filename):
    base_path = Path(__file__).parent
    file_path = str((base_path / "../../.." / filename).resolve())
    model_input = file_path.replace(os.sep, "/")

    mdf_model = load_mdf(model_input)
    if "Translated" in model_input:
        pytorch_model = mdf_to_pytorch(
            mdf_model, model_input, eval_models=False, version="mdf.0"
        )
    else:
        pytorch_model = mdf_to_pytorch(
            mdf_model, model_input, eval_models=False, version="mdf.s"
        )


if __name__ == "__main__":
    filename = "examples/MDF/ABCD.json"
    main()
