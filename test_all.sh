#!/bin/bash
set -ex

pip install .

# Note this:
#    1) runs examples to regenerate yaml/json...
#    2) tests examples with simple_scheduler
pytest -v -ra tests/interfaces/pytorch/*py
pytest -v -ra tests/*py

cd examples/MDF

## Test generating MDF models, saving json/yaml & running the models

python simple.py -run
python abcd.py -run
python arrays.py -run
python states.py -run -nogui
python abc_conditions.py -run

## Test exporting to NeuroML

python -m modeci_mdf.interfaces.neuroml.importer Simple.json -run
python -m modeci_mdf.interfaces.neuroml.importer ABCD.json -run
python -m modeci_mdf.interfaces.neuroml.importer States.json -run


## Test exporting to graphs via GraphViz

python -m modeci_mdf.interfaces.graphviz.importer Simple.json 1 -noview
mv simple_example.gv.png images/simple.png
python -m modeci_mdf.interfaces.graphviz.importer Simple.json 3 -noview
mv simple_example.gv.png images/simple_3.png
python -m modeci_mdf.interfaces.graphviz.importer ABCD.json 1 -noview
mv abcd_example.gv.png images/abcd.png
python -m modeci_mdf.interfaces.graphviz.importer ABCD.json 3 -noview
mv abcd_example.gv.png images/abcd_3.png
python -m modeci_mdf.interfaces.graphviz.importer Arrays.json 3 -noview
mv array_example.gv.png images/arrays.png
python -m modeci_mdf.interfaces.graphviz.importer States.yaml 3 -noview
mv state_example.gv.png images/states.png
python -m modeci_mdf.interfaces.graphviz.importer abc_conditions.yaml 3 -noview
mv abc_conditions_example.gv.png images/abc_conditions.png

## Test regenerating NeuroML

cd ../../examples/NeuroML
./regenerateAndTest.sh

## Test ONNX examples

cd ../../examples/ONNX
python simple_ab.py -run
python simple_abc.py
python simple_abcd.py

## Test ACT-R examples

cd ../../examples/ACT-R
python count.py
python addition.py

## Generate the docs

cd ../../docs
python generate.py
cd -
