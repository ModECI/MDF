#!/bin/bash
set -ex

# Script to regenerate and test the examples

cd MNIST

python keras_model.py -nogui

python keras_to_MDF.py -nogui

cd ../IRIS

python keras_model.py

python keras_to_MDF.py -nogui
