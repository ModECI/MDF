
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

    # parser = argparse.ArgumentParser(description=' Running the translator to stateful parameters')
    # parser.add_argument('--dt', default=5e-05, type=float,  help='time increment')
    # parser.add_argument('--run', default=False, type=bool,  help='Run the graph')



    # args = parser.parse_args()
    # print(args)
    dt = 0.01
    file_path = 'sample_test.json'
    data = convert_states_to_stateful_parameters(file_path, dt)
    # print(data)
    with open('Translated_'+ file_path, 'w') as fp:
        json.dump(data, fp,  indent=4)


    if "-run" in sys.argv:

        f = open(file_path)
        data = json.load(f)
        filtered_list = ['parameters','functions', 'states','output_ports','input_ports', 'notes']
        all_nodes = []
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
                        if (isinstance(vv, dict) and kk in filtered_list) or (isinstance(vv, str) and kk in filtered_list):
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

        expression_dict = {}
        def get_expression(d: Dict[str, Any] = None):
            """get any expression (including time derivative) for each state or output port variable
            Args:
                d: Node level dictionary with filtered keys
            Returns:
                store expression for each state or output port variable
            """
            for key in d.keys():
                vi = []
                li = []
                temp_dic = {}
                if 'states' in d[key].keys():
                    for state in d[key]['states'].keys():
                        
                        if 'time_derivative' in d[key]['states'][state].keys():
                            li.append(state+"#state#time#derivative")
                            vi.append(d[key]['states'][state]['time_derivative'])
                        elif any(x in d[key]['states'][state]['value'] for x in expression_items):
                            li.append(state+"#state#expression")
                            vi.append(d[key]['states'][state]['value'])
                        else:
                            li.append(state+"#state")
                            vi.append(None)
                if 'output_ports' in d[key].keys():
                    for output_port in d[key]['output_ports'].keys():
                        
                        if any(x in d[key]['output_ports'][output_port]['value'] for x in expression_items):
                            li.append(output_port+"#output#expression")
                            vi.append(d[key]['output_ports'][output_port]['value'])
                        else:
                            li.append(output_port+"#output")
                            vi.append(None)
                for i in range(len(vi)):
                    temp_dic[li[i]] = vi[i]
                expression_dict[key] = temp_dic
        get_expression(nodes_dict)

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
        
        mod_graph_old = load_mdf(file_path).graphs[0]
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
            t = eg.enodes['sine_node'].evaluable_stateful_parameters['time'].curr_value
            times.append(eg.enodes['sine_node'].evaluable_stateful_parameters['time'].curr_value)
            eg.evaluate()
            # times.append(t)            
            
            s.append(eg.enodes['sine_node'].evaluable_outputs['out_port1'].curr_value)
            if t == 0:
                eg_old.evaluate() # replace with initialize?
            else:
                eg_old.evaluate(time_increment=dt)
            s_old.append(eg_old.enodes['sine_node'].evaluable_outputs['out_port1'].curr_value)
            
            
        print(s_old[:10], s[:10], times[:10])

        print("Old json outputs>>>",eg_old.enodes['C'].evaluable_outputs['output_1'].curr_value, eg_old.enodes['A'].evaluable_outputs['output_1'].curr_value)
            
        print("Translated json output>>>",eg.enodes['C'].evaluable_outputs['output_1'].curr_value, eg.enodes['A'].evaluable_outputs['output_1'].curr_value)
            
      
        import matplotlib.pyplot as plt
        plt.plot(times,s)
        

        plt.show()
        plt.savefig('translated_sampletest_levelrate_sineplot.jpg')



if __name__ == "__main__":
    main()

