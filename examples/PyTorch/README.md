# PyTorch and MDF

The current implementation of our PyTorch to MDF conversion functionality is built
on top of the TorchScript infrastructure provided by PyTorch. PyTorch models that
can be translated to TorchScript (via `torch.jit.script` or `torch.jit.trace`) should
then be able to be converted to their MDF representation automatically. Below are
several working examples of this functionality.

### Inception Blocks Model

![Inception](inception.svg?raw=1)

To run an example of converting a PyTorch InceptionV3 like model written in PyTorch
to its MDF representation simply run:

```bash
python inception.py
```

This will define the model in PyTorch, invoke the TorchScript tracing compiler,
convert the underlying IR representation of the model to MDF. The MDF for this
model is the written to [inception.json](inception.json). The model is then executed
via the MDF scheduler and the results are compared to the native execution in PyTorch.

### Drift Diffusion Model

A simple example of a Drift Diffusion Model implemented in PyTorch and converted to
MDF can be executed with:

```bash
python pytorch_ddm.python
```

This will generate the MDF representation in [ddm.json](ddm.json)

### Multi-Layer Perceptron MDF to PyTorch Conversion
To run an example where a simple Multi-Layer Perceptron (MLP) created using the MDF specification is translated to a PyTorch model and executed using sample digit-recognition data, run:

```bash
python mlp.py
```
