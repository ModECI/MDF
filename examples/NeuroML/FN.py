from neuromllite import Network, Cell, Population, Synapse, RectangularRegion, RandomLayout
from neuromllite import Projection, RandomConnectivity, OneToOneConnector, Simulation, InputSource, Input

from neuromllite.NetworkGenerator import check_to_generate_or_run

import sys


def generate():

    dt = 0.05
    simtime = 100

    ################################################################################
    ###   Build new network

    net = Network(id='FN')
    net.notes = 'FitzHugh Nagumo cell model - originally specified in NeuroML/LEMS'

    net.parameters = { 'initial_w': 0.0,
                       'initial_v': -1,
                       'a_v': -0.3333333333333333,
                       'b_v': 0.0,
                       'c_v': 1.0,
                       'd_v': 1,
                       'e_v': -1.0,
                       'f_v': 1.0,
                       'time_constant_v': 1.0,
                       'a_w': 1.0,
                       'b_w': -0.8,
                       'c_w': 0.7,
                       'time_constant_w': 12.5,
                       'threshold': -1.0,
                       'mode': 1.0,
                       'uncorrelated_activity': 0.0,
                       'Iext': 0 }

    cellInput = Cell(id='fn',
                     lems_source_file='FN_Definitions.xml',
                     parameters={})
    for p in net.parameters:
        cellInput.parameters[p]=p
    net.cells.append(cellInput)


    r1 = RectangularRegion(id='region1', x=0,y=0,z=0,width=1000,height=100,depth=1000)
    net.regions.append(r1)


    pop = Population(id='FNpop',
                    size='1',
                    component=cellInput.id,
                    properties={'color':'0.2 0.2 0.2', 'radius':3},
                    random_layout = RandomLayout(region=r1.id))
    net.populations.append(pop)




    new_file = net.to_json_file('%s.json'%net.id)


    ################################################################################
    ###   Build Simulation object & save as JSON

    sim = Simulation(id='Sim%s'%net.id,
                     network=new_file,
                     duration=simtime,
                     dt=dt,
                     seed= 123,
                     recordVariables={'V':{'all':'*'},'W':{'all':'*'}},
                     plots2D={'VW':{'x_axis':'%s/0/fn/V'%pop.id,
                                 'y_axis':'%s/0/fn/W'%pop.id}})

    sim.to_json_file()

    return sim, net



if __name__ == "__main__":


    sim, net = generate()

    ################################################################################
    ###   Run in some simulators

    import sys

    check_to_generate_or_run(sys.argv, sim)
