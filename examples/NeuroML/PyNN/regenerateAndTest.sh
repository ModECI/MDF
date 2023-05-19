#!/bin/bash
set -ex

run_gui_examples=true

if [[ ($# -eq 1) && ($1 == '-nogui') ]]; then
    run_gui_examples=false
fi

####  Generate single cell example
python Generate.py -one -nml
pynml -validate OneCell.net.nml

python Generate.py -one -mdf
python RunInMDF.py OneCell.mdf.json -nogui


####  Generate multiple input example
python Generate.py -input_weights -nml
pynml -validate InputWeights.net.nml 

python Generate.py -input_weights -mdf
python RunInMDF.py InputWeights.mdf.json -nogui

