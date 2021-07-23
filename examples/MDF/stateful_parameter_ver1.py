"""
    Example of ModECI MDF - Testing state variables
"""

from modeci_mdf.mdf import *
import sys


def main():
    mod = Model(id="New_Stateful_Parameters")
    mod_graph = Graph(id="stateful_parameter_example")
    mod.graphs.append(mod_graph)

   

    ## Sine node...
    sine_node = Node(id="sine_node", parameters={"amp": 3, "dt": 0.1})

    s1 = Stateful_Parameter(id="time", default_initial_value=0, value="update_time")
    sine_node.stateful_parameters.append(s1)
    
    f1 = Function(
        id="update_time",
        function="linear",
        args={"variable0": s1.id, "slope": 1, "intercept": "dt"},
    )
    f2 = Function(
        id="sin_function",
        function="sin",
        args={"variable0": s1.id, "scale": "amp"},
    )


    sine_node.functions.append(f1)
    sine_node.functions.append(f2)



    op1 = OutputPort(id="out_port", value="sin_function")
    sine_node.output_ports.append(op1)

    mod_graph.nodes.append(sine_node)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        verbose = True
        
        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.execution_engine import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 0.1
        
        duration= 20
        t = 0
        recorded = {}
        times = []
        s = []
        while t<=duration:
            
            print("======   Evaluating at t = %s  ======"%(t))
           
            times.append(round(float(eg.enodes['sine_node'].evaluable_stateful_parameters['time'].curr_value),1))
            
            eg.evaluate() 
            s.append(eg.enodes['sine_node'].evaluable_outputs['out_port'].curr_value)
            
            # eg.evaluate() 
            t+=dt
        print(s, times)
        import matplotlib.pyplot as plt
        # plt.figure(figsize=(20,10))
        plt.plot(times,s)
        plt.show()
        plt.savefig('timestateful_sine_plot.jpg')
    return mod_graph


if __name__ == "__main__":
    main()
