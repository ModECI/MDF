#!/bin/bash
set -ex

## MDF to Pytorch

cd MDF_PyTorch
python MDF_to_PyTorch.py -test

cd ..
python mlp_pure_mdf.py -graph


python simple_pytorch_to_mdf.py -graph -graph-torch

python inception.py -graph

#python pytorch_ddm.py -graph

cd PyTorch_MDF/benchmark_script
python benchmark.py squeezenet1_1 count 10
cd ../..
