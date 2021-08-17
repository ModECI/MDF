from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import Node, Edge, OutputPort

from modeci_mdf.execution_engine import EvaluableGraph

from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW
import sys
import numpy as np

verbose = True
verbose = False

def execute(multi=False):

    mdf_model = load_mdf('FN.mdf.yaml')
    mod_graph = mdf_model.graphs[0]

    dt = 0.00005
    duration= 0.1

    if not multi:

        fn_node = mod_graph.nodes[0]
        fn_node.get_parameter('initial_v').value=[-1.]
        fn_node.get_parameter('initial_w').value=[0.]
        input = np.array([0])

    else:
        size = 15
        max_amp = 0.5
        input = np.array([ max_amp*(-1 + 2* i/size) for i in range(size+1)])
        #input = [-0.4,-0.2, 0.,0.2,0.4]
        input_node = Node(id="input_node", parameters={"input_level": input})

        op1 = OutputPort(id="out_port", value="input_level")
        input_node.output_ports.append(op1)
        mod_graph.nodes.append(input_node)

        fn_node = mod_graph.nodes[0]
        fn_node.get_parameter('initial_v').value=np.array([1.]*len(input))
        fn_node.get_parameter('initial_w').value=np.array([0.]*len(input))

        print(fn_node)

        e1 = Edge(
            id="input_edge",
            sender=input_node.id,
            sender_port=op1.id,
            receiver='FNpop_0',
            receiver_port='INPUT',
        )

        mod_graph.edges.append(e1)

        mdf_model.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="FNmulti",
            only_warn_on_fail=True  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )

        duration= 0.1

    eg = EvaluableGraph(mod_graph, verbose)
    #duration= 2
    t = 0

    times = []
    vv = {}
    ww = {}

    format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY

    while t<duration+0.00005:
        times.append(t)
        print("======   Evaluating at t = %s  ======"%(t))
        if t == 0:
            eg.evaluate(array_format=format) # replace with initialize?
        else:
            eg.evaluate(array_format=format, time_increment=dt)

        for i in range(len(eg.enodes['FNpop_0'].evaluable_parameters['V'].curr_value)):
            if not i in vv:
                vv[i]=[]
                ww[i]=[]
            v = eg.enodes['FNpop_0'].evaluable_parameters['V'].curr_value[i]
            w = eg.enodes['FNpop_0'].evaluable_parameters['W'].curr_value[i]
            vv[i].append(v)
            ww[i].append(w)
            if i==0:
                print('    Value at %s: v=%s, w=%s'%(t,v,w))
        t+=dt

    import matplotlib.pyplot as plt
    for vi in vv:
        plt.plot(times,vv[vi],label='V %.3f'%input[vi])
        plt.plot(times,ww[vi],label='W %.3f'%input[vi])
    plt.legend()

    if not multi:
        plt.savefig('MDFFNrun.png', bbox_inches='tight')

    if not "-nogui" in sys.argv:
        plt.show()


if __name__ == "__main__":

    execute("-multi" in sys.argv)
