#!/bin/bash
set -ex

python setup.py install

cd examples

python simple.py
python abcd.py

cd ..

python -m modeci_mdf.mdf
python -m modeci_mdf.simple_scheduler examples/Simple.json
python -m modeci_mdf.simple_scheduler examples/Simple.yaml
python -m modeci_mdf.simple_scheduler examples/ABCD.json
python -m modeci_mdf.simple_scheduler examples/ABCD.yaml


cd examples

python -m modeci_mdf.export.neuroml Simple.json
python -m modeci_mdf.export.neuroml ABCD.json
#python -m modeci_mdf.export.neuroml examples/ABCD.json -run

python -m modeci_mdf.export.graphviz Simple.json 3 -noview
python -m modeci_mdf.export.graphviz ABCD.json 1 -noview

cd ../docs
python generate.py
cd -
