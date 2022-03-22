# PyTorch and MDF

The current implementation of our PyTorch to MDF conversion functionality is built
on top of the TorchScript infrastructure provided by PyTorch. PyTorch models that
can be translated to TorchScript (via `torch.jit.script` or `torch.jit.trace`) should
then be able to be converted to their MDF representation automatically. Below are
several working examples of this functionality.

### Inception Blocks Model

![Inception from PyTorch](inception.svg?raw=1)
<img alt="Inception MDF" height="500" src="inception.png"/>

To run an example of converting a PyTorch InceptionV3 like model written in PyTorch
to its MDF representation simply run:

```bash
python inception.py
```

This will define the model in PyTorch, invoke the TorchScript tracing compiler,
convert the underlying IR representation of the model to MDF. The MDF for this
model is the written to [inception.json](inception.json). The model is then executed
via the MDF scheduler and the results are compared to the native execution in PyTorch.

The graph representation of the MDF model can be generated with:

```bash
python inception.py -graph
```

### Multi-Layer Perceptron MDF to PyTorch Conversion

To run an example where a simple Multi-Layer Perceptron (MLP) created using the MDF specification and executed using sample digit-recognition data, run:

```bash
python mlp_pure_mdf.py
```

A graph of the network can be created with `python mlp_pure_mdf.py -graph`:

<p align="center"><img src="mlp_pure_mdf.png" alt="mlp_pure_mdf.png" height="400"></p>


### MDF to PyTorch Conversion

To perform an MDF to PyTorch conversion, provide an MDF model as an input to the `mdf_to_pytorch` function
which is available in [exporter.py](https://github.com/ModECI/MDF/blob/development/src/modeci_mdf/interfaces/pytorch/exporter.py). The output of `mdf_to_pytorch`
are PyTorch models.  Below are some working examples of this functionality. The converted
models are available in folder: [MDF_PyTorch](https://github.com/ModECI/MDF/tree/development/examples/PyTorch/MDF_PyTorch).

The demo to convert an MDF model to PyTorch is at [MDF_to_PyTorch.py](https://github.com/ModECI/MDF/blob/development/examples/PyTorch/MDF_PyTorch/MDF_to_PyTorch.py).

Any model created using the MDF specification is translated to a PyTorch model, run:

```bash
python MDF_to_PyTorch
```

One of sample MDF examples [ABCD.json](../MDF/ABCD.json) is converted PyTorch [ABCD_pytorch.py](MDF_PyTorch/ABCD_pytorch.py)
The PyTorch model is further converted to ONNX [ABCD.onnx](MDF_PyTorch/ABCD.onnx) and the results are compared in all three environments.
