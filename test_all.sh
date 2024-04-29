#!/bin/bash
set -ex

pip install .[all]

# Note this:
#    1) runs examples to regenerate yaml/json...
#    2) tests examples with simple_scheduler
python -m pytest -ra tests/*.py
python -m pytest -ra tests/interfaces/onnx/*.py
python -m pytest -ra tests/interfaces/pytorch/test_export.py
#python -m pytest -ra tests/interfaces/pytorch/test_import.py # inception taking v long

cd examples/MDF

## Test generating MDF models, saving json/yaml & running the models

python simple.py -run
python abcd.py -run
python arrays.py -run
python states.py -run -nogui
python abc_conditions.py -run
python params_funcs.py -run

## Test exporting to NeuroML

python -m modeci_mdf.interfaces.neuroml.exporter Simple.json -run
python -m modeci_mdf.interfaces.neuroml.exporter ABCD.json -run
python -m modeci_mdf.interfaces.neuroml.exporter States.json -run


## Test exporting to graphs via GraphViz

python -m modeci_mdf.interfaces.graphviz.exporter Simple.json 1 -noview -horizontal
mv simple_example.gv.png images/simple.png
python -m modeci_mdf.interfaces.graphviz.exporter Simple.json 3 -noview
mv simple_example.gv.png images/simple_3.png
python -m modeci_mdf.interfaces.graphviz.exporter ABCD.json 1 -noview -horizontal
mv abcd_example.gv.png images/abcd.png
python -m modeci_mdf.interfaces.graphviz.exporter ABCD.json 3 -noview
mv abcd_example.gv.png images/abcd_3.png
python -m modeci_mdf.interfaces.graphviz.exporter Arrays.json 3 -noview
mv array_example.gv.png images/arrays.png
python -m modeci_mdf.interfaces.graphviz.exporter States.yaml 3 -noview
mv state_example.gv.png images/states.png
python -m modeci_mdf.interfaces.graphviz.exporter abc_conditions.yaml 3 -noview
mv abc_conditions_example.gv.png images/abc_conditions.png
python -m modeci_mdf.interfaces.graphviz.exporter ParametersFunctions.yaml 3 -noview
mv params_funcs_example.gv.png images/params_funcs.png

cd conditions
python everyNCalls.py -graph
mv everyncalls.png images/everyncalls.png
python timeInterval.py -graph
mv timeinterval.png images/timeinterval.png
python threshold.py -graph
mv threshold.png images/threshold.png
python composite_condition_example.py -graph
mv composite_example.png images/composite_example.png
cd ..

## Test regenerating NeuroML

cd RNN
./regenerate.sh


## Test regenerating NeuroML

cd ../../NeuroML
./regenerateAndTest.sh -nogui

## Test PyTorch examples

cd ../PyTorch
./regenerate.sh

## Test ONNX examples

cd ../ONNX
python simple_ab.py -run
python simple_abc.py
python simple_abcd.py
python abc_basic.py

## Test ACT-R examples

cd ../ACT-R
python count.py
python addition.py

## Test Keras examples

cd ../TensorFlow/Keras
./regenerate.sh

## Generate the docs

cd ../../../docs
python generate.py
cd ..

pre-commit run --all-files
