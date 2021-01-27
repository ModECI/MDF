#!/bin/bash
set -ex

python setup.py install

cd examples

python Simple.py
python ABCD.py

cd ..

cd modeci_mdf

python MDF.py
python SimpleScheduler.py
python SimpleScheduler.py ../examples/ABCD.json

cd export

python NeuroML.py
python NeuroML.py ../../examples/ABCD.json
