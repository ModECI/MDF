

from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import Node, Edge, OutputPort

from modeci_mdf.scheduler import EvaluableGraph

verbose = True
verbose = False

def execute():

    mod_graph = load_mdf('FN_extends.mdf.yaml').graphs[0]

    eg = EvaluableGraph(mod_graph, verbose)
    dt = 0.00005
    duration= 0.1
    #duration= 2
    t = 0

    times = []
    vv = {}
    ww = {}
    pops = ['FN_prototype']
    for pop in pops:
        vv[pop]=[]
        ww[pop]=[]

    while t<duration+dt:
        times.append(t)
        print("======   Evaluating at t = %s  ======"%(t))
        if t == 0:
            eg.evaluate() # replace with initialize?
        else:
            eg.evaluate(time_increment=dt)

        for pop in pops:
            v = eg.enodes[pop].evaluable_states['V'].curr_value
            w = eg.enodes[pop].evaluable_states['W'].curr_value
            vv[pop].append(v)
            ww[pop].append(w)
        t+=dt


    import matplotlib.pyplot as plt
    for pop in vv:
        #print('Pop: %s, %s, %s'%(pop, times, vv[pop]))
        plt.plot(times,vv[pop],label='V %s'%(pop))
        plt.plot(times,ww[pop],label='W %s'%(pop))
    plt.show()


if __name__ == "__main__":

    execute()
