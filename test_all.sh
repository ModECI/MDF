#!/bin/bash
set -ex

pip install .

# Note this:
#    1) runs examples to regenerate yaml/json...
#    2) tests examples with simple_scheduler
pytest


cd examples/MDF

python -m modeci_mdf.interfaces.neuroml.importer Simple.json -run
python -m modeci_mdf.interfaces.neuroml.importer ABCD.json -run
python -m modeci_mdf.interfaces.neuroml.importer States.json -run


python -m modeci_mdf.interfaces.graphviz.importer Simple.json 1 -noview
mv simple_example.gv.png simple.png


python -m modeci_mdf.interfaces.graphviz.importer Simple.json 3 -noview
python -m modeci_mdf.interfaces.graphviz.importer ABCD.json 1 -noview

python -m modeci_mdf.interfaces.graphviz.importer Arrays.json 3 -noview
python -m modeci_mdf.interfaces.graphviz.importer States.yaml 3 -noview

cd ../../docs
python generate.py
cd -
