
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

    dt = 5e-05
    file_path = 'FN.mdf.json'
    data, expression_dict, arg_dict = convert_states_to_stateful_parameters('../'+file_path, dt)
    # print(data)
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
        

        duration= 0.1
        t = 0
        recorded = {}
        times = []
        s = []
        vv = []
        ww = []

        vv_old = []
        ww_old = []
        
        while t<=duration + dt:
            print("======   Evaluating at t = %s  ======"%(t))
            

            

            vv.append(float(eg.enodes['FNpop_0'].evaluable_stateful_parameters['V'].curr_value))
            ww.append(float(eg.enodes['FNpop_0'].evaluable_stateful_parameters['W'].curr_value))
            
            
            # levels.append(eg.enodes['sine_node'].evaluable_stateful_parameters['level'].curr_value) 

           
            
            # print("time first>>>",type(t))
            t = float(eg.enodes['FNpop_0'].evaluable_stateful_parameters['time'].curr_value)
            times.append(float(eg.enodes['FNpop_0'].evaluable_stateful_parameters['time'].curr_value))
            
            if t == 0:
                eg_old.evaluate() # replace with initialize?
            else:
                eg_old.evaluate(time_increment=dt)

            vv_old.append(float(eg_old.enodes['FNpop_0'].evaluable_states['V'].curr_value))
            ww_old.append(float(eg_old.enodes['FNpop_0'].evaluable_states['W'].curr_value))
            
            eg.evaluate()
            
            

        
            
        print("Translated file W and V>>>",ww[:10],vv[:10])

        print("Old file W and V>>>",ww_old[:10],vv_old[:10])
        
        import matplotlib.pyplot as plt
        plt.plot(times,vv,label='V')
        plt.plot(times,ww,label='W')
        plt.legend()
        plt.show()
        plt.savefig('translated_FN_stateful_vw_plot.jpg')


if __name__ == "__main__":
    main()

