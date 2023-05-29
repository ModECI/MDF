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


####  Generate HH example
python Generate.py -hh -nml
pynml -validate HH.net.nml

python Generate.py -hh -mdf
python RunInMDF.py HH.mdf.json -nogui


####  Generate multiple input example
python Generate.py -input_weights -nml
pynml -validate InputWeights.net.nml

python Generate.py -input_weights -mdf
python RunInMDF.py InputWeights.mdf.json -nogui


####  Generate simple net example
python Generate.py -simple_net -nml
pynml -validate SimpleNet.net.nml

python Generate.py -simple_net -mdf
python RunInMDF.py SimpleNet.mdf.json -nogui


####  Generate bigger net example
python Generate.py -net1 -nml
pynml -validate Net1.net.nml

python Generate.py -net1 -mdf
python RunInMDF.py Net1.mdf.json -nogui
