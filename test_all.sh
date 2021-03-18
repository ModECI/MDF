#!/bin/bash
set -ex

python setup.py install

cd examples

python simple.py
python abcd.py

cd ..

python -m modeci_mdf.mdf
python -m modeci_mdf.simple_scheduler examples/ABCD.json

python -m modeci_mdf.export.neuroml examples/ABCD.json
#python -m modeci_mdf.export.neuroml examples/ABCD.json -run

python -m modeci_mdf.export.graphviz examples/Simple.json 1 -noview
python -m modeci_mdf.export.graphviz examples/ABCD.json 1 -noview
