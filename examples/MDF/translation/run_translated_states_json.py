
import json
import ntpath

from modeci_mdf.standard_functions import mdf_functions, create_python_expression, _add_mdf_function
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import *
from modeci_mdf.full_translator import *
from modeci_mdf.execution_engine import EvaluableGraph

import argparse
import sys

def main():

   
    dt = 0.01
    file_path = 'States.json'
    data, expression_dict, arg_dict  = convert_states_to_stateful_parameters('../'+file_path, dt)
    
    with open('Translated_'+ file_path, 'w') as fp:
        json.dump(data, fp,  indent=4)


    if "-run" in sys.argv:

        
        for node, keys in expression_dict.items():
            for key in keys.keys():
                if ("#state#time#derivative" in key) and (expression_dict[node][key] is not None):
                    _add_mdf_function("evaluate_{}_{}_next_value".format(node, key.split('#')[0]),
                                      description="computing the next value of stateful parameter {}".format(key.split('#')[0]),
                                      arguments=arg_dict[node], expression_string=str(key.split('#')[0]) + "+" "(dt*" + str(
                            expression_dict[node][key]) + ")", )
                
                elif ("#state#expression" in key) and (expression_dict[node][key] is not None):

                    _add_mdf_function("evaluate_{}_{}_next_value".format(node, key.split('#')[0]),
                                      description="computing the next value of stateful parameter {}".format(key.split('#')[0]),
                                      arguments=arg_dict[node], expression_string=expression_dict[node][key])

                elif ("#output#expression" in key) and (expression_dict[node][key] is not None):

                    _add_mdf_function("evaluate_{}_{}_value".format(node, key.split('#')[0]),
                                      description="computing the value of output port {}".format(key.split('#')[0]),
                                      arguments=arg_dict[node], expression_string=expression_dict[node][key])


                else:

                    print('No need to create MDF function for node %s, key %s since there is no expression!' % (
                        node, key))

        verbose = True
                
            
        mod_graph = load_mdf('Translated_%s'% file_path).graphs[0]
        eg = EvaluableGraph(mod_graph, verbose)
        
        mod_graph_old = load_mdf('../'+file_path).graphs[0]
        eg_old = EvaluableGraph(mod_graph_old, verbose)
        

        duration= 2
        t = 0
        recorded = {}
        times = []
        s = []
        s_old=[]
        while t<=duration:

           
            print("======   Evaluating at t = %s  ======"%(t))
            
            

            # levels.append(eg.enodes['sine_node'].evaluable_stateful_parameters['level'].curr_value) 
            # t+=args.dt
        
            
            
            # print("time first>>>",type(t))
            t = float(eg.enodes['sine_node'].evaluable_stateful_parameters['time'].curr_value)
            times.append(float(eg.enodes['sine_node'].evaluable_stateful_parameters['time'].curr_value))
            eg.evaluate()
            # times.append(t)            
            
            s.append(eg.enodes['sine_node'].evaluable_outputs['out_port'].curr_value)
            if t == 0:
                eg_old.evaluate() # replace with initialize?
            else:
                eg_old.evaluate(time_increment=dt)
            s_old.append(eg_old.enodes['sine_node'].evaluable_outputs['out_port'].curr_value)
            
            
        print(s_old[:10], s[:10], times[:10])
        import matplotlib.pyplot as plt
        plt.plot(times,s)
        

        plt.show()
        plt.savefig('translated_levelrate_sineplot.jpg')



if __name__ == "__main__":
    main()

