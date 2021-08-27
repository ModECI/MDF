#!/bin/bash

set -ex

run_gui_examples=true

if [[ ($# -eq 1) && ($1 == '-nogui') ]]; then
    run_gui_examples=false
fi

####  Generate and run jNeuroML version of the network from NeuroMLlite definition
python ABCD.py -jnml
####  Test running the jNeuroML version standalone (using https://github.com/NeuroML/pyNeuroML)
pynml LEMS_SimABCD.xml -nogui

####  Generate PsyNeuLink version of the network from NeuroMLlite definition
#python ABCD.py -pnl  # Generated MDF no longer valid...
####  Load in PsyNeuLink version & run


## Todo: fix failing!
##### python test_bids_import.py


####  Generate MDF version of the network from NeuroMLlite definition
python ABCD.py -mdf

####  Generate graph from MDF version
python -m modeci_mdf.interfaces.graphviz.importer ABCD.mdf.yaml 1 -noview

####  Test evaluating MDF version
python -m modeci_mdf.execution_engine ABCD.mdf.json

####  Generate a graph depicting the structure & *dynamics* of the network from the LEMS description
pynml LEMS_SimABCD.xml -lems-graph

if [ "$run_gui_examples" == true ]; then
    ####  Generate a graph depicting the structure of network from NeuroMLlite
    python ABCD.py -graph2

fi


####  Generate and run jNeuroML version of the network from NeuroMLlite definition
python FN.py -jnml
####  Generate PsyNeuLink version of the network from NeuroMLlite definition
#python FN.py -pnl  # Generated BIDS-MDF/PNL no longer valid...

####  Generate a graph depicting the structure of the network from the LEMS description
pynml LEMS_SimFN.xml -lems-graph

####  Generate MDF version of the network from NeuroMLlite definition
python FN.py -mdf

####  Generate graph from MDF version
python -m modeci_mdf.interfaces.graphviz.importer FN.mdf.yaml 3 -noview

####  Test evaluating MDF version
python -m modeci_mdf.execution_engine FN.mdf.json

####  Test running MDF version & save result
python FNrun.py -nogui

echo "Successfully ran all tests"
