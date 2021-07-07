

from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import Node, Edge, OutputPort

from modeci_mdf.execution_engine import EvaluableGraph

verbose = True
verbose = False

def execute(multi=False):

    mod_graph = load_mdf('FN.mdf.yaml').graphs[0]

    if not multi:

        fn_node = mod_graph.nodes[0]
        fn_node.parameters['initial_v']=[1.]
        fn_node.parameters['initial_w']=[0.]

    else:
        input = [-0.4,-0.2, 0.,0.2,0.4]
        input_node = Node(id="input_node", parameters={"input_level": input})

        op1 = OutputPort(id="out_port", value="input_level")
        input_node.output_ports.append(op1)
        mod_graph.nodes.append(input_node)

        fn_node = mod_graph.nodes[0]
        fn_node.parameters['initial_v']=[1.]*len(input)
        fn_node.parameters['initial_w']=[0.]*len(input)

        print(fn_node)

        e1 = Edge(
            id="input_edge",
            sender=input_node.id,
            sender_port=op1.id,
            receiver='FNpop_0',
            receiver_port='INPUT',
        )

        mod_graph.edges.append(e1)

    eg = EvaluableGraph(mod_graph, verbose)
    dt = 0.00005
    duration= 0.1
    #duration= 2
    t = 0

    times = []
    vv = {}
    ww = {}

    while t<duration+0.00005:
        times.append(t)
        print("======   Evaluating at t = %s  ======"%(t))
        if t == 0:
            eg.evaluate() # replace with initialize?
        else:
            eg.evaluate(time_increment=dt)

        for i in range(len(eg.enodes['FNpop_0'].evaluable_states['V'].curr_value)):
            if not i in vv:
                vv[i]=[]
                ww[i]=[]
            v = eg.enodes['FNpop_0'].evaluable_states['V'].curr_value[i]
            w = eg.enodes['FNpop_0'].evaluable_states['W'].curr_value[i]
            vv[i].append(v)
            ww[i].append(w)
        t+=dt

    import matplotlib.pyplot as plt
    for vi in vv:
        plt.plot(times,vv[vi],label='V_%s'%vi)
        plt.plot(times,ww[vi],label='W_%s'%vi)
    plt.show()


if __name__ == "__main__":

    execute()
