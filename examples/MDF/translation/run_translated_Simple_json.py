
import json
import ntpath

from modeci_mdf.standard_functions import mdf_functions, create_python_expression, _add_mdf_function
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import *
from modeci_mdf.full_translator import *
from modeci_mdf.execution_engine import EvaluableGraph
from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW
import argparse
import sys

def main():

    
   
    file_path = 'Simple.json'
    data = convert_states_to_stateful_parameters('../'+file_path)
    # print(data)
    with open('Translated_'+ file_path, 'w') as fp:
        json.dump(data, fp,  indent=4)


    if "-run" in sys.argv:

        
        
        verbose = True
                
            
        mod_graph = load_mdf('Translated_%s'% file_path).graphs[0]
        eg = EvaluableGraph(mod_graph, verbose)
        
        mod_graph_old = load_mdf('../'+file_path).graphs[0]
        eg_old = EvaluableGraph(mod_graph_old, verbose)
        
        
        
    
        format = FORMAT_NUMPY
        
           
        eg.evaluate(array_format=format)

        eg_old.evaluate(array_format=format)

        print("New file output value>>>",eg.enodes['processing_node'].evaluable_outputs['output_1'].curr_value)

        print("Old file output value>>>",eg_old.enodes['processing_node'].evaluable_outputs['output_1'].curr_value)
        
      

       



if __name__ == "__main__":
    main()

