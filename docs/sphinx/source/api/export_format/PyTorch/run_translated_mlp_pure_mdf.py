import json
import ntpath

from modeci_mdf.functions.standard import mdf_functions, create_python_expression
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import *
from modeci_mdf.full_translator import *
from modeci_mdf.execution_engine import EvaluableGraph

import argparse
import sys
import numpy as np
import os
import h5py
import time


def main():

    verbose = True
    dt = 5e-05
    file_path = "mlp_pure_mdf.json"
    data = convert_states_to_stateful_parameters(file_path, dt)
    # print(data)
    with open("Translated_" + file_path, "w") as fp:
        json.dump(data, fp, indent=4)

    test_all = "-test" in sys.argv

    mod_graph = load_mdf("Translated_%s" % file_path).graphs[0]

    # mdf_to_graphviz(mod_graph,view_on_render=not test_all, level=3)

    from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

    format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY

    eg = EvaluableGraph(mod_graph, verbose=False)
    eg.evaluate(array_format=format)

    print("Finished evaluating graph using array format %s" % format)

    for n in [
        "mlp_input_layer",
        "mlp_relu_1",
        "mlp_hidden_layer_with_relu",
        "mlp_output_layer",
    ]:
        out = eg.enodes[n].evaluable_outputs["out_port"].curr_value
        print(f"Final output value of node {n}: {out}, shape: {out.shape}")

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=2,
            filename_root="mlp_pure_mdf",
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )

    if test_all:
        # Iterate on training data, feed forward and log accuracy
        imgs = np.load("example_data/imgs.npy")
        labels = np.load("example_data/labels.npy")

        import torch.nn

        matches = 0
        imgs_to_test = imgs[:300]

        start = time.time()
        for i in range(len(imgs_to_test)):
            ii = imgs[i, :, :]
            target = labels[i]
            img = torch.Tensor(ii).view(-1, 14 * 14).numpy()
            # plot_img(img, 'Post_%i (%s)'%(i, img.shape))
            print(
                "***********\nTesting image %i (label: %s): %s\n%s"
                % (i, target, np.array2string(img, threshold=5, edgeitems=2), img.shape)
            )
            # print(mod_graph.nodes[0].parameters['input'])
            mod_graph.nodes[0].get_parameter("input").value = img
            eg = EvaluableGraph(mod_graph, verbose=False)
            eg.evaluate(array_format=format)
            for n in ["mlp_output_layer"]:
                out = eg.enodes[n].evaluable_outputs["out_port"].curr_value
                print(
                    "Output of evaluated graph: %s %s (%s)"
                    % (out, out.shape, type(out).__name__)
                )
                prediction = np.argmax(out)

            match = target == int(prediction)
            if match:
                matches += 1
            print(f"Target: {target}, prediction: {prediction}, match: {match}")
        t = time.time() - start
        print(
            "Matches: %i/%i, accuracy: %s%%. Total time: %.4f sec (%.4fs per run)"
            % (
                matches,
                len(imgs_to_test),
                (100.0 * matches) / len(imgs_to_test),
                t,
                t / len(imgs_to_test),
            )
        )


if __name__ == "__main__":
    main()
