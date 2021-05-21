import os
import sys
import h5py
import importlib
from collections import defaultdict
from inspect import getmembers, signature, getsource, isclass

import numpy as np

import torch
import torch.nn as nn

from modeci_mdf.interfaces.pytorch import mod_torch_builtins as torch_builtins

from modeci_mdf.utils import load_mdf
from modeci_mdf.scheduler import EvaluableGraph


def generate_initializer_call(func_class, params, idx=False):

	settable_params = get_instance_params(func_class)

	text = ""

	for param in settable_params:
		if param in params:

			param_text = "nn.Parameter(torch.Tensor({}))".format(params[param])

			if not idx:
				text += "\n\t\tself.function.{} = {}".format(param, param_text)
			else:
				text += "\n\t\tself.function_list[-1].{} = {}".format(param, param_text)

	return text


def get_instance_params(funcname):

	# Want to make a dummy instance to introspect
	sig = signature(funcname)
	args = []
	for param in sig.parameters.values():
		if "=" not in str(param):
			arg_type = param.annotation
			if arg_type == int:
				args.append(1)
			elif arg_type == list:
				args.append([1])
			elif arg_type == torch.Tensor:
				args.append(torch.Tensor([1]))
	dummy = funcname(*args)

	params = []

	for member in getmembers(dummy):
		name, member_type = member
		if type(member_type) == nn.parameter.Parameter:
			params.append(name)
	del dummy

	return params


def get_module_declaration_text(name, node_dict, execution_order, declared_module_types):

	declaration_text = ("\nclass {}(nn.Module):"
						"\n\tdef __init__(self):"
						"\n\t\tsuper().__init__()"
						"\n\t\tself.calls = 0"
						).format(name)

	functions = node_dict["functions"]
	parameters = node_dict["parameters"]

	# Single function node
	if len(functions) == 1:

		current_function = functions[0]
		function_name = current_function.id
		function_type = current_function.function
		function_args = current_function.args


		"""
		Expand alias_table and move to separate module
		"""
		alias_table = {
		  "matmul":"matmul",
		  "conv_2d":"conv2d",
		  "add":"add",
		  "argmax":"argmax"
		}

		function_type_alias = str(function_type).lower()

		if function_type_alias in alias_table:
			function_type = alias_table[function_type_alias]

		# For torch builtins implemented as modules
		if function_type in torch_builtins.__all__:
			function_object = getattr(torch_builtins, function_type)

			# Grab source code and prepend to text
			if function_type not in declared_module_types:
				declaration_text = "\n" + getsource(function_object) + declaration_text
				declared_module_types.add(function_type)
			declaration_text += "\n\t\tself.function = {}()".format(function_type)


			# Get the signature to call this module
			source_string = getsource(function_object)
			args = source_string.split("forward(")[1].split("):")[0].split(", ")[1:]
			#annotated_args = ["{}:{}".format(k, function_args[k]) for k in args]
			#call_signature = "{}({})".format(function_name, ",".join(args))

			constructor_info = ("builtin", function_name, function_type, args)

			# Make a forward call
			declaration_text += "\n\tdef forward(self, {}):".format(",".join(args))
			declaration_text += "\n\t\treturn self.function({})".format(",".join(args))

		else:
			# Get torch.nn module
			function_object = None
			for class_tup in getmembers(nn, isclass):
				if class_tup[0].lower()==function_type.lower():
					function_type = class_tup[0]
					function_object = class_tup[1]
					break
			constructor_info = ("nn", function_name, function_type, [])

			declaration_text += "\n\t\tself.function = nn.{}()".format(str(function_object).split(".")[-1].split("'")[0])

			# TODO: Resolve ordering of args
			args = list(function_args.keys())
			declaration_text += "\n\tdef forward(self, {}):".format(",".join(args))
			declaration_text += "\n\t\treturn self.function({})".format(",".join(args))

	# TODO: Fix for Multi function nodes

	return declaration_text, constructor_info


def generate_main_forward(nodes, execution_order, constructor_calls):
	#
	# for node in nodes:
	#     print(node.id)

	node_dict = {node.id:node for node in nodes}

	# Index intermediate variables
	std_var_idx = 0
	nstd_var_idx = 0

	# Map intermediate variable name to the module that produced it
	return_vars = defaultdict(list)

	main_forward = "\n\tdef forward(self, input):"

	# TODO: Handle multi-input graphs
	for idx, node_name in enumerate(execution_order):
		if idx==0:
			standard_arg = "input"
		else:
			standard_arg = "svar_{}".format(std_var_idx-1)

		node = node_dict[node_name]
		non_standard_args = []
		if node.parameters:
			for param_key in list(node.parameters.keys()):
				# TODO: Resolve ordering
				pre_expression = "\n\t\tnsvar_{} = torch.Tensor({})".format(nstd_var_idx, node.parameters[param_key])
				main_forward += pre_expression
				non_standard_args.append("nsvar_{}".format(nstd_var_idx))
				nstd_var_idx+=1

		args = [standard_arg]
		args.extend(non_standard_args)

		expression = "\n\t\tsvar_{} = self.{}({})".format(std_var_idx, node_name, ",".join(args))
		main_forward += expression
		std_var_idx+=1
	main_forward+="\n\t\treturn {}".format("svar_{}".format(std_var_idx-1))


	return main_forward


def build_script(nodes, execution_order, conditions=None):
	"""
	Create and assemble following components for a complete script:

		* Module declarations
			* Initialization of functions
			* Definition of forward function

		* Model main call declaration:
			* Initialization of subcomponents
			* Forward function logic
	"""
	script = ""
	imports_string = ("import torch"
					  "\nimport torch.nn as nn")

	# Declarations string
	modules_declaration_text = ""
	constructor_calls = {}
	declared_module_types = set()


	for node in nodes:

		id, funcs, params = node.id, node.functions, node.parameters
		node_dict = {"functions":funcs, "parameters":params}

		declaration_text, constructor_call = get_module_declaration_text(id,node_dict,execution_order,declared_module_types)

		modules_declaration_text += declaration_text
		constructor_calls[id] = constructor_call

	# Build Main call
	main_call_declaration = ("\nclass Model(nn.Module):"
							"\n\tdef __init__(self):"
							"\n\t\tsuper().__init__()")


	for node in execution_order:
		main_call_declaration += "\n\t\tself.{} = {}()".format(node, node)

	# Build Main forward
	main_call_forward = generate_main_forward(nodes, execution_order, constructor_calls)

	# Compose script
	script += imports_string
	script += modules_declaration_text
	script += main_call_declaration
	script += main_call_forward

	script += "\nmodel = Model()"
	return script



def _generate_scripts_from_json(model_input):
	"""
	Parse elements from MDF and use text_tools module to make script
	"""
	file_name = model_input.split("/")[-1].split(".")[0]
	file_dir = "/".join(model_input.split("/")[:-1])

	model = load_mdf(model_input)
	scripts = {}

	for graph in model.graphs:
		nodes = graph.nodes
		# Read weights.h5 if exists
		if "weights.h5" in os.listdir(file_dir):
			weight_dict = h5py.File(os.path.join(file_dir, "weights.h5"), 'r')

			# Hack to fix problem with HDF5 parameters
			for node in graph.nodes:
				if node.parameters:
					for param_key, param_val in node.parameters.items():
						if param_key in ["weight", "bias"] and type(param_val)==str:
							# Load and reassign
							array = weight_dict[param_val][:]
							np.set_printoptions(threshold=sys.maxsize)
							node.parameters[param_key] = np.array2string(array, separator=", ")

		evaluable_graph = EvaluableGraph(graph, False)
		#root = evaluable_graph.root_nodes[0]
		enodes = evaluable_graph.enodes
		edges = evaluable_graph.ordered_edges
		try:
			conditions = evaluable_graph.conditions
		except AttributeError:
			conditions = {}

		# Use edges and nodes to construct execution order
		execution_order = []
		for idx, edge in enumerate(edges):
			if idx==0:
				execution_order.append(edge.sender)
			execution_order.append(edge.receiver)

		# Build script
		script = build_script(nodes, execution_order, conditions=conditions)
		scripts[graph.id] = script

	return scripts

def _script_to_model(script):

	import importlib.util

	#For testing, need to add prefix if calling from out of examples directory
	module_path = os.path.join(os.getcwd(), *sys.argv[0].split("/")[:-1], "module.py")

	with open(module_path, mode="w") as f:
		f.write(script)

	torch_spec = importlib.util.spec_from_file_location("module", module_path)
	torch_module = importlib.util.module_from_spec(torch_spec)
	torch_spec.loader.exec_module(torch_module)

	model = torch_module.model

	os.remove(module_path)

	return model

def mdf_to_pytorch(model_input, eval_models=True):
	"""
	Load and return all models specified in an MDF graph
	"""
	scripts = _generate_scripts_from_json(model_input)
	models = {}

	for script_name, script in scripts.items():
		model = _script_to_model(script)

		if eval_models:
			model.eval()

		models[script_name] = model

	return models

__all__ = ["mdf_to_pytorch"]
