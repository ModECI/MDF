
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

	
  
	file_path = 'abc_conditions.json'
	data, expression_dict, arg_dict = convert_states_to_stateful_parameters('../'+file_path)
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
		
		
	
		format = FORMAT_NUMPY
		
		   
		eg.evaluate(array_format=format)

		eg_old.evaluate(array_format=format)

		
		print("Old file output value>>>",eg.enodes['C'].evaluable_outputs['output_1'].curr_value)

		print("New file output value>>>",eg_old.enodes['C'].evaluable_outputs['output_1'].curr_value)
		

	   



if __name__ == "__main__":
	main()

