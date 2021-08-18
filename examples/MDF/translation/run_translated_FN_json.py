
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
    data = convert_states_to_stateful_parameters('../'+file_path, dt)
    # print(data)
    with open('Translated_'+ file_path, 'w') as fp:
        json.dump(data, fp,  indent=4)


    if "-run" in sys.argv:

        
        

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
            

            

            vv.append(float(eg.enodes['FNpop_0'].evaluable_parameters['V'].curr_value))
            ww.append(float(eg.enodes['FNpop_0'].evaluable_parameters['W'].curr_value))
            
            
            # levels.append(eg.enodes['sine_node'].evaluable_stateful_parameters['level'].curr_value) 

           
            
            # print("time first>>>",type(t))
            t = float(eg.enodes['FNpop_0'].evaluable_parameters['time'].curr_value)
            times.append(t)
            
            if t == 0:

                eg_old.evaluate() # replace with initialize?
            else:
                
                eg_old.evaluate(time_increment=dt)

            vv_old.append(eg_old.enodes['FNpop_0'].evaluable_parameters['V'].curr_value)
            ww_old.append(eg_old.enodes['FNpop_0'].evaluable_parameters['W'].curr_value)
            
            eg.evaluate()
        
            

        print("Times>>>", times[:10])        
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

