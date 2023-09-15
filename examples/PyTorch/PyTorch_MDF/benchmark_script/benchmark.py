import time
import torch
import sys
import os
import json
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf
from modeci_mdf.execution_engine import EvaluableGraph
import importlib.util

MDF_ENGINE = "MDF"
PYTORCH_ENGINE = "PyTorch"


# This crawls over the directory and identifies available and viable models
# to be used in the benchmarking app.
# Example: Run "python benchmark.py" in your terminal to see available models.
# ***Note*** remove the Run keyword and quotation marks from the above before
# running the example above.
def get_model_names(directory):
    excluded_files = ["benchmark.py"]
    model_names = []

    for file in os.listdir(directory):
        if file.endswith(".py") and file not in excluded_files:
            file_path = os.path.join(directory, file)
            module_name = os.path.splitext(file)[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_example_input = getattr(module, "get_example_input", None)
            get_pytorch_model = getattr(module, "get_pytorch_model", None)

            if get_example_input and get_pytorch_model:
                model_names.append(module_name)

    return sorted(model_names)


# This block handles the instantiation of the type of model you are interested in,
# it loads in necessary resources such as the model definition, a suitable dataset
# and also instantiates all of them, preparing them for use.
# This block also handles looking into the model script, checking if necessary resources
# and definitions are available. If reverse is the case, it displays a message displaying
# how to modify the files and get these resource available globally.
if len(sys.argv) >= 2 and "--all" not in sys.argv and "-run" not in sys.argv:
    model_name = sys.argv[1]
    file_path = os.path.join("..", f"{model_name}.py")
    module_name = os.path.splitext(file_path)[0]
    if model_name not in get_model_names(".."):
        print("Please include your model type")
        print("Usage: python benchmark.py [model] count <integer>")
        print("Example: python benchmark.py convolution count 10")
        print(
            "***Note***: The larger the count the greater the run time, keep count within 10 - 50 range"
        )
        # sys.exit(1)
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_example_input = getattr(module, "get_example_input", None)
        get_pytorch_model = getattr(module, "get_pytorch_model", None)

        if get_example_input and get_pytorch_model:
            data = get_example_input()
            model = get_pytorch_model()
            model_type = model.__class__.__name__

        if not get_pytorch_model:
            print("The function get_pytorch_model() seems to be absent in the file.")
            print(
                "Modify the file to be benchmarked as such:\n"
                "def get_pytorch_model():\n    model = model_definition_goes_here\n    return model"
            )
            # sys.exit(1)

        if not get_example_input:
            print("The function get_example_input() seems to be absent in the file.")
            print(
                "Modify the file to be benchmarked as such:\n"
                "def get_example_input():\n    x = data_to_be_predicted_goes_here"
            )
            # sys.exit(1)
    except ImportError:
        print(
            f"Could not import  '{model_name}'. Make sure the model definition exists and is properly defined."
        )
        # sys.exit(1)


# This handles the number of iterations, it also handles other subtle processes such as if
# the keyword count is omitted or if the count value is not given or if count is incorrectly spelt.
# It also checks if an integer count value is passed.
if (
    "count" not in sys.argv
    and len(sys.argv) >= 2
    and "--all" not in sys.argv
    and "-run" not in sys.argv
):
    print('You seem to be forgetting to include keyword:"count"')
    print("Usage: python benchmark.py [model] count <integer>")
    print("Example: python benchmark.py convolution count 10")
    print(
        "***Note***: The larger the count the greater the run time, keep count within 10 - 50 range"
    )
    # sys.exit(1)

try:
    if len(sys.argv) >= 3 and "--all" not in sys.argv and "-run" not in sys.argv:
        count_index = sys.argv.index("count")
        if count_index + 1 < len(sys.argv):
            count = sys.argv[count_index + 1]
            if count.isdigit():
                count = int(count)
            else:
                print("Invalid count argument. It should be an integer.")
                # sys.exit(1)
        else:
            raise ValueError("No count argument provided.")
except ValueError:
    print("Please input a correct count of any positive integer")
    print("Example: python benchmark.py -convolution count 10")
    print(
        "***Note***: The larger the count the greater the run time, keep count within 10 - 50 range"
    )
    # sys.exit(0)


# This takes the shape of the data collected from the model data definition and generates similarly shaped
# datasets all containing randomized values within their tensors for every iteration. This function
# returns a list the same datasets to pass through the benchmark engine for pytorch and mdf ensuring
# both are given similar data.
def data_random_gen(count, data):
    return [torch.rand_like(data) for _ in range(count)]


# This take model definitions, such as model definition data, model definition and selects the
# type of benchmark to be done. It outputs the prediction time and also the prediction count.
def benchmark_engine(model, model_name, data, engine):
    total_time = 0.0
    prediction_count = 0

    if engine == MDF_ENGINE:
        mdf_model, params_dict = pytorch_to_mdf(model=model, args=(data[0]), trace=True)
        mdf_graph = mdf_model.graphs[0]
        eg = EvaluableGraph(graph=mdf_graph, verbose=False)
        node_density = len(mdf_graph.__getattribute__("nodes"))

    for d in data:
        print(
            "\n  =====  Running model: %s on engine %s, count %i/%i"
            % (model_name, engine, prediction_count + 1, len(data))
        )
        start_time = time.time()
        if engine == PYTORCH_ENGINE:
            with torch.no_grad():
                pred = model(d)
            pred = pred.argmax().item()
        elif engine == MDF_ENGINE:
            params_dict["input1"] = d.detach().numpy()
            eg.evaluate(initializer=params_dict)
            mdf_pred = eg.output_enodes[0].get_output()
            pred = mdf_pred.argmax().item()
        end_time = time.time()

        total_time += end_time - start_time
        prediction_count += 1
    if engine == MDF_ENGINE:
        return total_time, prediction_count, node_density
    return total_time, prediction_count


# This displays the PYTORCH work after a successful run,
# if the run is unsuccessful, the word is not displayed
# similar to the other words such as TO and MDF.
def print_pytorch_word():
    pytorch_word = [
        "   ****   *     *  *******   ***    ****       ***  *     * ",
        "   *   *   *   *      *    *     *  *   *    *      *     * ",
        "   ****     ***       *    *     *  *****   *       * *** * ",
        "   *         *        *    *     *  *    *   *      *     * ",
        "   *         *        *      ***    *     *    ***  *     * ",
    ]

    for line in pytorch_word:
        print(line)


def print_to_word():
    to_word = [
        " *******   ***  ",
        "    *    *     *",
        "    *    *     *",
        "    *    *     *",
        "    *      *** ",
    ]

    for line in to_word:
        print(line)


def print_MDF_word():
    pytorch_word = [
        "   **     **  ****    ***** ",
        "   * *   * *  *    *  *     ",
        "   *   *   *  *    *  ***** ",
        "   *       *  *    *  *     ",
        "   *       *  ****    *     ",
    ]

    for line in pytorch_word:
        print(line)


# This momentarily outputs a graphical instance of the output of the selected model,
# displaying their count and prediction time.
if len(sys.argv) > 2 and "--all" not in sys.argv and "-run" not in sys.argv:
    data = data_random_gen(count, data)


# This triggers the benchmark app, displaying all the available and viable models
# in the directory. Use the example above to get a birds eye view of all the available
# models and further useage examples.
if len(sys.argv) >= 1 and "--all" not in sys.argv and "-run" in sys.argv:
    available_models = get_model_names("..")
    print("Please provide a model name")
    print("Usage: python benchmark.py [model] count <integer>")
    print("Example: python benchmark.py convolution count 10")
    print("Available models:")
    for model_name in available_models:
        print(f"    {model_name}")

    for model_name in available_models:
        if model_name == "convolution":
            file_path = os.path.join("..", f"{model_name}.py")
            module_name = os.path.splitext(file_path)[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_example_input = getattr(module, "get_example_input", None)
            get_pytorch_model = getattr(module, "get_pytorch_model", None)
            model = get_pytorch_model()
            data = get_example_input()
            model_type = model.__class__.__name__
            count = 100
            data = data_random_gen(count, data)
            pytorch_time, pytorch_predictions = benchmark_engine(
                model, model_name, data, PYTORCH_ENGINE
            )
            mdf_time, mdf_predictions, node_density = benchmark_engine(
                model, model_name, data, MDF_ENGINE
            )
            results = []
            result_entry = {
                "model_name": model_name,
                "model_type": model_type,
                "pytorch_time": pytorch_time,
                "pytorch_predictions": pytorch_predictions,
                "mdf_time": mdf_time,
                "mdf_predictions": mdf_predictions,
                "node density": node_density,
                "mdf/pytorch ratio": "%.2f" % (mdf_time / pytorch_time),
            }
            results.append(result_entry)

            with open(f"{model_name}_benchmark.json", "w") as json_file:
                json.dump(results, json_file, indent=4)


def main():

    pytorch_time, pytorch_predictions = benchmark_engine(
        model, model_name, data, PYTORCH_ENGINE
    )
    mdf_time, mdf_predictions, node_density = benchmark_engine(
        model, model_name, data, MDF_ENGINE
    )

    print_pytorch_word()
    print("\n")
    print_to_word()
    print("\n")
    print_MDF_word()

    pytorch_text = f"It takes PyTorch {pytorch_time:.4f} seconds to make {pytorch_predictions} predictions with the {model_type} model"
    mdf_text = f"It takes MDF {mdf_time:.4f} seconds to make {mdf_predictions} predictions with the {model_type} model"

    concat_text = f"{pytorch_text}\n\n{mdf_text}"
    print(concat_text)

    results = []
    result_entry = {
        "model_name": model_name,
        "model_type": model_type,
        "pytorch_time": round(pytorch_time, 4),
        "pytorch_predictions": pytorch_predictions,
        "mdf_time": round(mdf_time, 4),
        "mdf_predictions": mdf_predictions,
        "node density": node_density,
        "mdf : pytorch ratio": "%.2f" % (mdf_time / pytorch_time),
    }
    results.append(result_entry)

    with open(f"{model_name}_benchmark.json", "w") as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":

    if "--all" in sys.argv:
        available_models = get_model_names("..")

        data_count = {
            "convolution": 10000,
            "simple_Convolution": 10000,
        }

        results = []
        for model_name in available_models:
            file_path = os.path.join("..", f"{model_name}.py")
            module_name = os.path.splitext(file_path)[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_example_input = getattr(module, "get_example_input", None)
            get_pytorch_model = getattr(module, "get_pytorch_model", None)
            model = get_pytorch_model()
            data = get_example_input()
            model_type = model.__class__.__name__
            if model_name in data_count.keys():
                count = data_count[model_name]
            else:
                count = 100

            data = data_random_gen(count, data)
            pytorch_time, pytorch_predictions = benchmark_engine(
                model, model_name, data, PYTORCH_ENGINE
            )
            mdf_time, mdf_predictions, node_density = benchmark_engine(
                model, model_name, data, MDF_ENGINE
            )

            result_entry = {
                "model_name": model_name,
                "model_type": model_type,
                "pytorch_time": round(pytorch_time, 4),
                "pytorch_predictions": pytorch_predictions,
                "mdf_time": round(mdf_time, 4),
                "mdf_predictions": mdf_predictions,
                "mdf : pytorch ratio": "%.2f" % (mdf_time / pytorch_time),
                "node density": node_density,
            }
            results.append(result_entry)

        print_pytorch_word()
        print("\n")
        print_to_word()
        print("\n")
        print_MDF_word()

        with open("benchmark_results.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

    elif len(sys.argv) >= 2 and "-run" not in sys.argv:
        main()
