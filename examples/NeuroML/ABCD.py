from neuromllite import Network, Cell, Population, Synapse, RectangularRegion, RandomLayout
from neuromllite import Projection, RandomConnectivity, OneToOneConnector, Simulation, InputSource, Input

from neuromllite.NetworkGenerator import check_to_generate_or_run

import sys


def generate():

    dt = 100 # ms, so 0.1s
    simtime = 5000 # ms, so 50s

    ################################################################################
    ###   Build new network

    net = Network(id='ABCD')
    net.notes = 'Example of a simplified network'

    net.parameters = { 'A_initial': 0.1, 'A_slope': 2.2}

    cellInput = Cell(id='a_input',
                     lems_source_file='PNL.xml',
                     parameters={'variable':'A_initial'})

    net.cells.append(cellInput)

    cellA = Cell(id='a', lems_source_file='PNL.xml')
    net.cells.append(cellA)
    cellB = Cell(id='b', lems_source_file='PNL.xml')
    net.cells.append(cellB)
    cellC = Cell(id='c', lems_source_file='PNL.xml')
    net.cells.append(cellC)
    cellD = Cell(id='d', lems_source_file='PNL.xml')
    net.cells.append(cellD)

    rsDL = Synapse(id='rsDL', lems_source_file='PNL.xml')
    net.synapses.append(rsDL)

    r1 = RectangularRegion(id='region1', x=0,y=0,z=0,width=1000,height=100,depth=1000)
    net.regions.append(r1)


    pAin = Population(id='A_input',
                    size='1',
                    component=cellInput.id,
                    properties={'color':'0.2 0.2 0.2', 'radius':3},
                    random_layout = RandomLayout(region=r1.id))
    net.populations.append(pAin)

    pA = Population(id='A',
                    size='1',
                    component=cellA.id,
                    properties={'color':'0 0.9 0', 'radius':5},
                    random_layout = RandomLayout(region=r1.id))
    net.populations.append(pA)

    pB = Population(id='B',
                    size='1',
                    component=cellB.id,
                    properties={'color':'.8 .8 .8', 'radius':5},
                    random_layout = RandomLayout(region=r1.id))
    net.populations.append(pB)

    pC = Population(id='C',
                    size='1',
                    component=cellC.id,
                    properties={'color':'0.7 0.7 0.7', 'radius':5},
                    random_layout = RandomLayout(region=r1.id))
    net.populations.append(pC)

    pD = Population(id='D',
                    size='1',
                    component=cellD.id,
                    properties={'color':'0.7 0 0', 'radius':5},
                    random_layout = RandomLayout(region=r1.id))
    net.populations.append(pD)

    silentDLin = Synapse(id='silentSyn_proj_input', lems_source_file='PNL.xml')
    net.synapses.append(silentDLin)
    net.projections.append(Projection(id='proj_input',
                                      presynaptic=pA.id,
                                      postsynaptic=pB.id,
                                      synapse=rsDL.id,
                                      pre_synapse=silentDLin.id,
                                      type='continuousProjection',
                                      weight=1,
                                      random_connectivity=RandomConnectivity(probability=1)))

    silentDL0 = Synapse(id='silentSyn_proj0', lems_source_file='PNL.xml')
    net.synapses.append(silentDL0)
    net.projections.append(Projection(id='proj0',
                                      presynaptic=pAin.id,
                                      postsynaptic=pA.id,
                                      synapse=rsDL.id,
                                      pre_synapse=silentDL0.id,
                                      type='continuousProjection',
                                      weight=1,
                                      random_connectivity=RandomConnectivity(probability=1)))

    silentDL1 = Synapse(id='silentSyn_proj1', lems_source_file='PNL.xml')
    net.synapses.append(silentDL1)
    net.projections.append(Projection(id='proj1',
                                      presynaptic=pA.id,
                                      postsynaptic=pC.id,
                                      synapse=rsDL.id,
                                      pre_synapse=silentDL1.id,
                                      type='continuousProjection',
                                      weight=1,
                                      random_connectivity=RandomConnectivity(probability=1)))

    silentDL2 = Synapse(id='silentSyn_proj2', lems_source_file='PNL.xml')
    net.synapses.append(silentDL2)
    net.projections.append(Projection(id='proj2',
                                      presynaptic=pB.id,
                                      postsynaptic=pD.id,
                                      synapse=rsDL.id,
                                      pre_synapse=silentDL2.id,
                                      type='continuousProjection',
                                      weight=1,
                                      random_connectivity=RandomConnectivity(probability=1)))

    silentDL3 = Synapse(id='silentSyn_proj3', lems_source_file='PNL.xml')
    net.synapses.append(silentDL3)
    net.projections.append(Projection(id='proj3',
                                      presynaptic=pC.id,
                                      postsynaptic=pD.id,
                                      synapse=rsDL.id,
                                      pre_synapse=silentDL3.id,
                                      type='continuousProjection',
                                      weight=1,
                                      random_connectivity=RandomConnectivity(probability=1)))
    

    new_file = net.to_json_file('%s.json'%net.id)


    ################################################################################
    ###   Build Simulation object & save as JSON

    sim = Simulation(id='Sim%s'%net.id,
                     network=new_file,
                     duration=simtime,
                     dt=dt,
                     seed= 123,
                     recordVariables={'OUTPUT':{'all':'*'}}) # ,'INPUT':{'all':'*'}

    sim.to_json_file()

    return sim, net



if __name__ == "__main__":


    sim, net = generate()

    ################################################################################
    ###   Run in some simulators

    import sys

    check_to_generate_or_run(sys.argv, sim)
