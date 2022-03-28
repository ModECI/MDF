#!/bin/bash
set -ex

python generate_rnn.py -graph
python generate_rnn.py -run -nogui

python generate_iaf.py -graph
python generate_iaf.py -run -nogui
