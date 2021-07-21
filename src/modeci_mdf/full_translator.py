import json
import ntpath
from modeci_mdf.standard_functions import mdf_functions, create_python_expression, _add_mdf_function
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from modeci_mdf.utils import load_mdf, print_summary

from modeci_mdf.execution_engine import EvaluableGraph

import glom
def convert_states_to_stateful_parameters(file_path: str=None, dt = 5e-05):

	"""Translates json file if with states to json file with stateful_parameters, otherwise unchanged
	Args:
		file_path: File in Json Format
	Returns:
		file in json format
	"""
	f = open(file_path)
	data = json.load(f)
	
	filtered_list = ['parameters','functions', 'states','output_ports','input_ports','notes']
	all_nodes = []
	all_keys = []
	def keysExtractor(nested_dictionary):
		for k, v in nested_dictionary.items():
			
			if isinstance(v, dict) and k in 'conditions':
				continue
			elif isinstance(v, dict):
				all_keys.append(k)
				if isinstance(v, dict) and k in 'nodes':
					break
				elif isinstance(v, dict):
					keysExtractor(v)
			
	keysExtractor(data)
	path='.'.join(all_keys)

	def nodeExtractor(nested_dictionary: Dict[str, Any] = None):
		"""Extracts all the node objects in the graph
		Args:
			nested_dictionary: input data
		Returns:
			Dictionary of node objects
		"""
		for k, v in nested_dictionary.items():
			if isinstance(v, dict) and k in 'nodes':
				all_nodes.append(v.keys())
			elif isinstance(v, dict):
				nodeExtractor(v)
	nodeExtractor(data)
	nodes_dict = dict.fromkeys(all_nodes[0])
	

	for key in list(nodes_dict.keys()):
		nodes_dict[key] = {}

	def parameterExtractor(nested_dictionary: Dict[str, Any] = None):
		""" Extracts Parameters, states, functions, input and output ports at each node object
		Args:
			nested_dictionary: Input Data
		Returns:
			stores states, parameters, functions, input and output ports
		"""
		for k, v in nested_dictionary.items():
			if isinstance(v, dict) and k in list(nodes_dict.keys()):
				for kk, vv in v.items():
					
					if (isinstance(vv, dict) and kk in filtered_list) or (isinstance(vv, str) and kk in filtered_list) :
						
						nodes_dict[k][kk] = vv
			if isinstance(v, dict):
				parameterExtractor(v)
	parameterExtractor(data)

	arg_dict = {}
	def get_arguments(d:Dict[str, Any] = None):
		""" Extracts all parameters including stateful,dt for each node object
		Args:
			d: Node level dictionary with filtered keys
		Returns:
			all parameters for each node object
		"""
		for key in d.keys():
			vi = []
			flag = 0
			if 'parameters' in d[key].keys():
				vi += list(d[key]['parameters'].keys())
			if 'states' in d[key].keys():
				vi += list(d[key]['states'].keys())
			if 'states' in d[key].keys():
				for state in d[key]['states'].keys():
					if 'time_derivative' in d[key]['states'][state].keys():
						flag = 1
					if flag == 1 and 'dt' not in vi :
						vi.append('dt')

			arg_dict[key] = vi
	get_arguments(nodes_dict)

	time_derivative_dict = {}
	def get_time_derivative(d: Dict[str, Any] = None):
		"""get time_derivative expression for each state variable
		Args:
			d: Node level dictionary with filtered keys
		Returns:
			store time_derivative expression for each state variable
		"""
		for key in d.keys():
			vi = []
			li = []
			temp_dic = {}
			if 'states' in d[key].keys():
				for state in d[key]['states'].keys():
					li.append(state)
					if 'time_derivative' in d[key]['states'][state].keys():
						vi.append(d[key]['states'][state]['time_derivative'])
					else:
						vi.append(None)
			for i in range(len(vi)):
				temp_dic[li[i]] = vi[i]
			time_derivative_dict[key] = temp_dic
	get_time_derivative(nodes_dict)

	
	def createFunctions(d: Dict[str, Any] = None):
		"""create functions for time_derivative expression for each state variable
		Args:
			d: Node level dictionary with filtered keys
		Returns:
			functions replacing time derivative for each state variable
		"""
		for key in d.keys():

			if 'states' in d[key].keys():
				statelist=[]
				d[key]['functions']={}
				for idx,state in enumerate(list(d[key]['states'].keys())):
					if 'time_derivative' in d[key]['states'][state].keys():
						d[key]['functions']["evaluated_{}_{}_next_value".format(key, state)]={}
						d[key]['functions']["evaluated_{}_{}_next_value".format(key, state)]['function']={}
						d[key]['functions']["evaluated_{}_{}_next_value".format(key, state)]['function']= "evaluate_{}_{}_next_value".format(key, state)
						d[key]['functions']["evaluated_{}_{}_next_value".format(key, state)]['args']=  {}
						for param in arg_dict[key]:
							d[key]['functions']["evaluated_{}_{}_next_value".format(key, state)]['args'][param] = param


					
					statelist.append(state)
					if idx>0:
						for prev_state in statelist[:-1]:

							d[key]['functions']["evaluated_{}_{}_next_value".format(key, state)]['args'][prev_state] = "evaluated_{}_{}_next_value".format(key, prev_state)


	createFunctions(nodes_dict)
	

	def changetoValue(d: Dict[str, Any] = None):
		"""Converts states into stateful_parameters, adds dt to parameters 
		Args:
			d: dictionary with states information at the Node
		Returns:
			dictionary with stateful_parameters information
		"""
		for key in d.keys():
			if 'states' in d[key].keys():
				for state in list(d[key]['states'].keys()):
					if 'time_derivative' in d[key]['states'][state].keys():
						d[key]['parameters']['dt']=dt
						if 'default_initial_value' in d[key]['states'][state].keys():
							if d[key]['states'][state]['default_initial_value'] in d[key]['parameters']:
								d[key]['states'][state]['default_initial_value'] = d[key]['parameters'][d[key]['states'][state]['default_initial_value']]
						else:
							d[key]['states'][state]['default_initial_value'] = 0

						d[key]['states'][state].pop('time_derivative')
						d[key]['states'][state]['value']="evaluated_{}_{}_next_value".format(key, state)
						d[key]['states']['time'] = {'default_initial_value': 0, 'value': 'time + dt'}
					elif 'default_initial_value' not in d[key]['states'][state].keys() :
						d[key]['states'][state]['default_initial_value'] = 0

		for key in d.keys():
			if 'states' in d[key].keys():
				d[key]['stateful_parameters'] = d[key].pop('states')
	changetoValue(nodes_dict)
	glom.assign(data,path,nodes_dict)
	def repl(dr):
		"""Replaces all names containing states into stateful_parameters 
		Args:
			dr: full dictionary with states name
		Returns:
			dictionary with stateful_parameters name
		"""
		dr=str(dr)
		dr = dr.replace('states', 'stateful_parameters')
		dr = dr.replace('States', 'Stateful_Parameters')
		dr = dr.replace('state_example', 'stateful_parameters_example')
		return eval(dr)
	data= repl(data)
	return data
	


  
	

