#!/bin/bash
set -ex

pip install .

# Note this:
#    1) runs examples to regenerate yaml/json...
#    2) tests examples with simple_scheduler
pytest

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
mv simple_example.gv.png simple.png
python -m modeci_mdf.interfaces.graphviz.importer Simple.json 3 -noview
mv simple_example.gv.png simple_3.png
python -m modeci_mdf.interfaces.graphviz.importer ABCD.json 1 -noview
mv abcd_example.gv.png abcd.png
python -m modeci_mdf.interfaces.graphviz.importer Arrays.json 3 -noview
mv array_example.gv.png arrays.png
python -m modeci_mdf.interfaces.graphviz.importer States.yaml 3 -noview
mv state_example.gv.png states.png
python -m modeci_mdf.interfaces.graphviz.importer abc_conditions.yaml 3 -noview
mv abc_conditions_example.gv.png abc_conditions.png


## Generate the docs

cd ../../docs
python generate.py
cd -
