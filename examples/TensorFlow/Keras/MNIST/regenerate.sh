#!/bin/bash
set -ex

python keras_model.py -nogui

python keras_to_MDF.py

cd Keras_to_MDF_IRIS

python keras_model.py

python keras_to_MDF.py
