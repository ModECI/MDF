#!/bin/bash
set -ex

pip install .

# Note this:
#    1) runs examples to regenerate yaml/json...
#    2) tests examples with simple_scheduler
pytest


python -m modeci_mdf.export.neuroml examples/Simple.json
python -m modeci_mdf.export.neuroml examples/Simple.json -run
python -m modeci_mdf.export.neuroml examples/ABCD.json
python -m modeci_mdf.export.neuroml examples/ABCD.json -run

cd examples

python -m modeci_mdf.export.graphviz Simple.json 1 -noview
mv simple_example.gv.png simple.png


python -m modeci_mdf.export.graphviz Simple.json 3 -noview
python -m modeci_mdf.export.graphviz ABCD.json 1 -noview

python -m modeci_mdf.export.graphviz Arrays.json 3 -noview
python -m modeci_mdf.export.graphviz States.yaml 3 -noview

cd ../docs
python generate.py
cd -
