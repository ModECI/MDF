#!/bin/bash
set -ex

python setup.py install

# Note this:
#    1) runs examples to regenerate yaml/json...
#    2) tests examples with simple_scheduler
pytest

cd examples

python -m modeci_mdf.export.neuroml Simple.json
python -m modeci_mdf.export.neuroml ABCD.json
#python -m modeci_mdf.export.neuroml examples/ABCD.json -run

python -m modeci_mdf.export.graphviz Simple.json 1 -noview
mv simple_example.gv.png simple.png


python -m modeci_mdf.export.graphviz Simple.json 3 -noview
python -m modeci_mdf.export.graphviz ABCD.json 1 -noview

python -m modeci_mdf.export.graphviz Arrays.json 3 -noview
python -m modeci_mdf.export.graphviz States.yaml 3 -noview

cd ../docs
python generate.py
cd -
