import numpy
import time
import torch
import sys
import warnings
import re
from PIL import Image, ImageDraw, ImageFont
import os

from modeci_mdf.interfaces.pytorch import pytorch_to_mdf
from modeci_mdf.execution_engine import EvaluableGraph


def get_model_names(directory):
    excluded_files = ["benchmark.py"]
    model_names = [
        os.path.splitext(file)[0]
        for file in os.listdir(directory)
        if file.endswith(".py") and file not in excluded_files
    ]
    return model_names


if len(sys.argv) == 1:
    available_models = get_model_names(".")
    print("Please provide a model name")
    print("Available models:")
    for model in available_models:
        print(f"    {model}")
    sys.exit(1)

if len(sys.argv) >= 1:
    model_name = sys.argv[1]
    if model_name not in get_model_names("."):
        print("Your seem to be forgetting to include your model type")
        print("Usage: python benchmark.py [model] count <integer>")
        print("Example: python benchmark.py -convolution count 10")
        print(
            "***Note***: The larger the count the greater the run time, keep count within 10 - 50 range"
        )
        sys.exit(1)
    try:
        module = __import__(model_name)
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
            sys.exit(1)

        if not get_example_input:
            print("The function get_example_input() seems to be absent in the file.")
            print(
                "Modify the file to be benchmarked as such:\n"
                "def get_example_input():\n    x = data_to_be_predicted_goes_here"
            )
            sys.exit(1)
    except ImportError:
        print(
            f"Could not import  '{model_name}'. Make sure the model definition exists and is properly defined."
        )
        sys.exit(1)

if "count" not in sys.argv and len(sys.argv) >= 2:
    print('Your seem to be forgetting to include keyword:"count"')
    print("Usage: python benchmark.py [model] count <integer>")
    print("Example: python benchmark.py -convolution count 10")
    print(
        "***Note***: The larger the count the greater the run time, keep count within 10 - 50 range"
    )
    sys.exit(1)

try:
    if len(sys.argv) >= 3:
        count_index = sys.argv.index("count")
        if count_index + 1 < len(sys.argv):
            count = sys.argv[count_index + 1]
            if count.isdigit():
                count = int(count)
            else:
                print("Invalid count argument. It should be an integer.")
                sys.exit(1)
        else:
            raise ValueError("No count argument provided.")
except ValueError:
    print("Please input a correct count of any positive integer")
    print("Example: python benchmark.py -convolution count 10")
    print(
        "***Note***: The larger the count the greater the run time, keep count within 10 - 50 range"
    )
    sys.exit(1)


def pytorch_benchmark(loop_count, model, data):
    total_pytorch_prediction_time = 0.0
    prediction_count = 0

    for i in range(loop_count):
        start_time = time.time()
        with torch.no_grad():
            pred = model(data)
        prediction = pred.argmax().item()
        end_time = time.time()
        total_pytorch_prediction_time += end_time - start_time
        prediction_count += 1
    return total_pytorch_prediction_time, prediction_count


def mdf_benchmark(loop_count, model, data):
    total_mdf_prediction_time = 0.0
    prediction_count = 0
    mdf_model, params_dict = pytorch_to_mdf(model=model, args=(data), trace=True)

    mdf_graph = mdf_model.graphs[0]
    eg = EvaluableGraph(graph=mdf_graph, verbose=False)
    warnings.resetwarnings()

    for i in range(loop_count):
        start_time = time.time()
        params_dict["input1"] = data.detach().numpy()
        eg.evaluate(initializer=params_dict)
        mdf_pred = eg.output_enodes[0].get_output()
        mdf_pred = mdf_pred.argmax().item()
        end_time = time.time()
        total_mdf_prediction_time += end_time - start_time
        prediction_count += 1

    return total_mdf_prediction_time, prediction_count


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


def write_text_to_png(text, filename):
    # Create an image with white background
    width, height = 2000, 1200  # Increased image size
    background_color = (255, 255, 255)  # White
    image = Image.new("RGB", (width, height), background_color)

    # Draw the text on the image
    draw = ImageDraw.Draw(image)
    font_size = 45  # Increased font size
    font = ImageFont.truetype("arial.ttf", font_size)  # Load a TrueType font
    text_color = (0, 0, 0)  # Black
    text_position = (20, 20)
    draw.text(text_position, text, font=font, fill=text_color)

    # Save the image as a PNG file
    png_filename = filename + ".png"
    image.save(png_filename, "PNG")

    # Open the PNG image using the default image viewer
    try:
        if sys.platform == "win32":
            os.startfile(png_filename)
        elif sys.platform == "darwin":
            subprocess.run(["open", png_filename], check=True)
        else:
            print("Unsupported platform for opening files.")
    except Exception as e:
        print("Error:", e)

    # Wait for the user to close the image viewer
    input("Press Enter after viewing the image...")

    # Delete the PNG file
    try:
        os.remove(png_filename)
        print("PNG file deleted.")
    except Exception as e:
        print("Error deleting the PNG file:", e)


def main():

    pytorch_time, pytorch_predictions = pytorch_benchmark(count, model, data)
    mdf_time, mdf_predictions = mdf_benchmark(count, model, data)
    warnings.resetwarnings()

    print_pytorch_word()
    print("\n")
    print_to_word()
    print("\n")
    print_MDF_word()

    pytorch_text = f"It takes PyTorch {pytorch_time:.4f} seconds to make {pytorch_predictions} predictions with the {model_type} model"
    mdf_text = f"It takes MDF {mdf_time:.4f} seconds to make {mdf_predictions} predictions with the {model_type} model"

    concat_text = f"{pytorch_text}\n\n{mdf_text}"
    write_text_to_png(concat_text, "benchmark")


if __name__ == "__main__":
    main()
