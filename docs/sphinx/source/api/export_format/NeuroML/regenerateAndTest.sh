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

####  Generate a graph depicting the structure & *dynamics* of the network from the LEMS description
pynml LEMS_SimABCD.xml -lems-graph

####  Generate a graph depicting the structure of network from NeuroMLlite
python ABCD.py -graph2 -nogui
mv ABCD.gv.png ABCD.nmllite.png


####  Generate MDF version of the network from NeuroMLlite definition
python ABCD.py -mdf

####  Generate graph from MDF version
python -m modeci_mdf.interfaces.graphviz.exporter ABCD.mdf.yaml 1 -noview
mv ABCD.gv.png ABCD.1.mdf.png
python -m modeci_mdf.interfaces.graphviz.exporter ABCD.mdf.yaml 3 -noview
mv ABCD.gv.png ABCD.mdf.png

####  Test evaluating MDF version
##python -m modeci_mdf.execution_engine ABCD.mdf.json



####  Generate and run jNeuroML version of the network from NeuroMLlite definition
python FN.py -jnml
####  Generate PsyNeuLink version of the network from NeuroMLlite definition
#python FN.py -pnl  # Generated BIDS-MDF/PNL no longer valid...

####  Generate a graph depicting the structure of the network from the LEMS description
pynml LEMS_SimFN.xml -lems-graph

####  Generate MDF version of the network from NeuroMLlite definition
python FN.py -mdf

####  Generate graph from MDF version
python -m modeci_mdf.interfaces.graphviz.exporter FN.mdf.yaml 3 -noview

####  Test evaluating MDF version
python -m modeci_mdf.execution_engine FN.mdf.json

####  Test running MDF version & save result
python FNrun.py -nogui



####  Generate and run jNeuroML version of the network from NeuroMLlite definition
python Izhikevich.py -jnml

####  Generate a graph depicting the structure of the network from the LEMS description
pynml LEMS_SimIzhikevichTest.xml -lems-graph

####  Generate MDF version of the network from NeuroMLlite definition
python Izhikevich.py -mdf

####  Generate graph from MDF version
python -m modeci_mdf.interfaces.graphviz.exporter IzhikevichTest.mdf.yaml 2 -noview

####  Test evaluating MDF version
python -m modeci_mdf.execution_engine IzhikevichTest.mdf.yaml

####  Test running MDF version & save result
python Izh_run.py -nogui


echo "Successfully ran all tests"
