from modeci_mdf.mdf import *

import numpy as np
import sys
import h5py
import time
import os

# Note: the weights for this model were precomputed and saved in the file weights.h5
def get_weight_info():

    weights = {}
    f = h5py.File("weights.h5", "r")

    for key in list(f.keys()):
        weight_mat = f[key][:]
        weights[key] = weight_mat
        print(f"Loaded {key}: {weight_mat.shape}")
    f.close()

    return weights


def get_model():

    mod = Model(id="mlp_pure_mdf")
    mod_graph = Graph(id=mod.id)
    mod.graphs.append(mod_graph)

    weights = get_weight_info()

    dim0 = 16
    weight = weights[
        "weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_input_layer.parameters.weight"
    ]
    bias = weights[
        "weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_input_layer.parameters.bias"
    ]

    dummy_input = np.zeros((1, 14 * 14))
    # dummy_input = np.ones((14*14))

    input_node = Node(id="mlp_input_layer")
    input_node.parameters.append(Parameter(id="input", value=dummy_input))
    input_node.parameters.append(Parameter(id="weight", value=weight.T))
    input_node.parameters.append(Parameter(id="bias", value=bias.T))

    f1 = Parameter(id="mul", function="MatMul", args={"A": "input", "B": "weight"})

    input_node.parameters.append(f1)

    f2 = Parameter(
        id="sum",
        function="linear",
        args={"variable0": "mul", "slope": 1, "intercept": "bias"},
    )
    input_node.parameters.append(f2)

    input_node.output_ports.append(OutputPort(id="out_port", value="sum"))
    mod_graph.nodes.append(input_node)

    relu1_node = Node(id="mlp_relu_1")
    relu1_node.input_ports.append(InputPort(id="in_port"))
    mod_graph.nodes.append(relu1_node)

    f1 = Parameter(id="relu1", function="Relu", args={"A": "in_port"})
    relu1_node.parameters.append(f1)

    relu1_node.output_ports.append(OutputPort(id="out_port", value=f1.id))

    e1 = Edge(
        id="edge_1",
        sender=input_node.id,
        sender_port=input_node.output_ports[0].id,
        receiver=relu1_node.id,
        receiver_port=relu1_node.input_ports[0].id,
    )
    mod_graph.edges.append(e1)

    weight = weights[
        "weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_hidden_layer_with_relu.parameters.weight"
    ]
    bias = weights[
        "weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_hidden_layer_with_relu.parameters.bias"
    ]

    hr_node = Node(id="mlp_hidden_layer_with_relu")

    hr_node.parameters.append(Parameter(id="weight", value=weight.T))
    hr_node.parameters.append(Parameter(id="bias", value=bias.T))

    mod_graph.nodes.append(hr_node)
    hr_node.input_ports.append(InputPort(id="in_port"))

    f1 = Parameter(id="mul", function="MatMul", args={"A": "in_port", "B": "weight"})
    hr_node.parameters.append(f1)

    f2 = Parameter(
        id="sum",
        function="linear",
        args={"variable0": "mul", "slope": 1, "intercept": "bias"},
    )
    hr_node.parameters.append(f2)

    f3 = Parameter(id="relu2", function="Relu", args={"A": "sum"})
    hr_node.parameters.append(f3)

    hr_node.output_ports.append(OutputPort(id="out_port", value="relu2"))

    e2 = Edge(
        id="edge_2",
        sender=relu1_node.id,
        sender_port=relu1_node.output_ports[0].id,
        receiver=hr_node.id,
        receiver_port=hr_node.input_ports[0].id,
    )
    mod_graph.edges.append(e2)

    weight = weights[
        "weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_output_layer.parameters.weight"
    ]
    bias = weights[
        "weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_output_layer.parameters.bias"
    ]

    out_node = Node(id="mlp_output_layer")

    out_node.parameters.append(Parameter(id="weight", value=weight.T))
    out_node.parameters.append(Parameter(id="bias", value=bias.T))

    mod_graph.nodes.append(out_node)
    out_node.input_ports.append(InputPort(id="in_port"))

    f1 = Parameter(id="mul", function="MatMul", args={"A": "in_port", "B": "weight"})
    out_node.parameters.append(f1)

    f2 = Parameter(
        id="sum",
        function="linear",
        args={"variable0": "mul", "slope": 1, "intercept": "bias"},
    )
    out_node.parameters.append(f2)

    out_node.output_ports.append(OutputPort(id="out_port", value="sum"))

    e3 = Edge(
        id="edge_3",
        sender=hr_node.id,
        sender_port=hr_node.output_ports[0].id,
        receiver=out_node.id,
        receiver_port=out_node.input_ports[0].id,
    )
    mod_graph.edges.append(e3)

    return mod


def main():

    test_all = "-test" in sys.argv

    mod = get_model()

    mod_graph = mod.graphs[0]

    yaml_file = mod.to_yaml_file("%s.yaml" % mod.id)
    json_file = mod.to_json_file("%s.json" % mod.id)

    if "-mdf_to_pytorch" in sys.argv:
        print("Exporting model to pure PyTorch")
        from modeci_mdf.interfaces.pytorch import mdf_to_pytorch

        pytorch_model = mdf_to_pytorch(
            mod, yaml_file, eval_models=False, version="mdf.s"
        )

    from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

    format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY

    from modeci_mdf.execution_engine import EvaluableGraph

    eg = EvaluableGraph(mod_graph, verbose=False)
    eg.evaluate(array_format=format)

    from modelspec.utils import _val_info

    print("Finished evaluating graph using array format %s" % format)

    for n in [
        "mlp_input_layer",
        "mlp_relu_1",
        "mlp_hidden_layer_with_relu",
        "mlp_output_layer",
    ]:
        out = _val_info(eg.enodes[n].evaluable_outputs["out_port"].curr_value)
        print(f"Final output value of node {n}:\t {out}")

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
        all_guess = None
        labelled_guess = {}

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

                """print(
                    "Guesses:  %s, %s"
                    % (all_guess, all_guess.shape if all_guess is not None else "-")
                )"""
                if all_guess is None:
                    all_guess = out
                else:
                    all_guess = np.concatenate((all_guess, out))

                if target not in labelled_guess:
                    labelled_guess[target] = out
                else:
                    labelled_guess[target] = np.concatenate(
                        (labelled_guess[target], out)
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

        print(f"Guesses:  {all_guess}, {all_guess.shape}")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        title = "Guesses"
        # plt.title(title)
        # fig.canvas.set_window_title(title)
        import matplotlib

        cm = matplotlib.cm.get_cmap("Blues")

        sorted_guesses = labelled_guess[0]
        for i in range(9):
            sorted_guesses = np.concatenate((sorted_guesses, labelled_guess[i + 1]))

        im = plt.imshow(sorted_guesses, cmap=cm, aspect="auto")

        cbar = plt.colorbar(im)

        ax = plt.gca()

        ax.set_xticks([i for i in range(10)])

        ax.set_xlabel("Est. likelihood of each digit")

        ax.set_ylabel("%i images ordered by label" % (len(imgs_to_test)))

        plt.savefig("mlp_pure_mdf.results.png", bbox_inches="tight")

        if "-nogui" not in sys.argv:
            plt.show()


if __name__ == "__main__":
    main()
