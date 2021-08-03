"""
    Example of ModECI MDF - Testing state variables
"""

from modeci_mdf.mdf import *
import sys


def main():
    mod = Model(id="New_Stateful_Parameters_ver2")
    mod_graph = Graph(id="stateful_parameter_example2")
    mod.graphs.append(mod_graph)

   

    ## Sine node...
    sine_node = Node(id="sine_node", parameters={"amp": 3, "period": 0.4, "dt":0.01, "two_pi":6.283185})
    # sine_node = Node(id="sine_node", parameters={"amp": 3, "combined": 0.01 * 6.283185/0.4,  "_combined": -0.01 * 6.283185/0.4 })

    
    s1 = Stateful_Parameter(id="level", default_initial_value=0, value="update_level")
    sine_node.stateful_parameters.append(s1)

    s2 = Stateful_Parameter(id="rate", default_initial_value=1, value="update_rate")
    sine_node.stateful_parameters.append(s2)


    f1 = Function(
        id="update_rate",
        function="linear",
        args={"variable0": s1.id, "slope": "-dt*two_pi/period", "intercept": s2.id},
    )

    f2 = Function(
        id="update_level",
        function="linear",
        args={"variable0": f1.id, "slope": "dt*two_pi/period", "intercept": s1.id},
    )
    
    
    
    

    sine_node.functions.append(f1)
    sine_node.functions.append(f2)



    op1 = OutputPort(id="out_port", value="amp * level")
    sine_node.output_ports.append(op1)

    mod_graph.nodes.append(sine_node)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        verbose = True
        
        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.scheduler import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 0.01
        
        duration= 2
        t = 0
        recorded = {}
        times = []
        s = []
        levels=[]
        while t<=duration:

           
            print("======   Evaluating at t = %s  ======"%(t))
           
            # levels.append(eg.enodes['sine_node'].evaluable_stateful_parameters['level'].curr_value) 
            eg.evaluate() 
            s.append(eg.enodes['sine_node'].evaluable_outputs['out_port'].curr_value)
            
         
            t+=dt
            times.append(t)
        print(s[:5],times[:5])
        import matplotlib.pyplot as plt
        plt.plot(times,s)
        plt.show()
        plt.savefig('levelratestateful_sine_plot.jpg')
    return mod_graph


if __name__ == "__main__":
    main()
