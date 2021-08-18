import json
import ntpath

from modeci_mdf.standard_functions import mdf_functions, create_python_expression, _add_mdf_function
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import *
from modeci_mdf.execution_engine import EvaluableGraph

import argparse
import sys

def main():

	file_path = 'test_function.json'
	if "-run" in sys.argv:
		verbose = True
				
			
	
		mod_graph = load_mdf(file_path).graphs[0]
		eg= EvaluableGraph(mod_graph, verbose)
		

		duration= 2
		t = 0
		recorded = {}
		times = []
		s = []
		
		dt = 0.1
		while t<=duration:

		   
			print("======   Evaluating at t = %s  ======"%(t))
			
			

			# levels.append(eg.enodes['sine_node'].evaluable_stateful_parameters['level'].curr_value) 
			# t+=args.dt
		
			
			
			# print("time first>>>",type(t))
			
			times.append(t)
			eg.evaluate()
			# times.append(t)            
			
			s.append(eg.enodes['counter_node'].evaluable_outputs['out_port'].curr_value)
			t+=dt
			
		print(s[:10], times[:10])
		import matplotlib.pyplot as plt
		plt.plot(times,s)
		

		plt.show()
		plt.savefig('testfunction_counterplot.jpg')



if __name__ == "__main__":
	main()

