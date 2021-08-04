import json
from modeci_mdf.standard_functions import mdf_functions, create_python_expression,_add_mdf_function
f = open('../States.json')
data = json.load(f)


filtered_list=['parameters','states']
all_nodes=[]

def parameterExtractor(nested_dictionary):
  for k,v in nested_dictionary.items():

    if isinstance(v,dict) and k in 'nodes':

      all_nodes.append(v.keys())

      # dic[v.keys()]={}
    elif isinstance(v, dict):
      # print('inside this>>>',k,v)
      parameterExtractor(v)

parameterExtractor(data)
dic =dict.fromkeys(all_nodes[0])
for key in list(dic.keys()):

  dic[key] = {}


def parameterExtractor(nested_dictionary,nodes=None):
  for k,v in nested_dictionary.items():

    if isinstance(v,dict) and k in list(dic.keys()):
      for kk,vv in v.items():
        if isinstance(vv,dict) and kk in filtered_list:

          dic[k][kk]=vv

    # if isinstance(v,dict) and k in filtered_list:

    #   dic[k]=v
    if isinstance(v, dict):
      # print('inside this>>>',k,v)
      parameterExtractor(v)
parameterExtractor(data)
f.close()
# print("dic after>>>",dic)
# first step 1 - amp, period ->  add to this the state variables like level and rate, and initiliaze with defualt
arg_dict={}

def get_arguments(d):


  for key in d.keys():
    vi=[]

    flag=0



    vi+=list(d[key]['parameters'].keys())
    vi+=list(d[key]['states'].keys())

    for state in d[key]['states'].keys():

      if 'time_derivative' in d[key]['states'][state].keys():
        flag=1
    if flag==1:
      vi.append('dt')
    # print(key,vi)
    arg_dict[key] = vi

get_arguments(dic)
# for key, value in dic.items():
    # arg_list+=list(value.keys())
    # arg_dict.update(value)
print("This arg_dict contains input to update mdf function", arg_dict)
# #second step 2 -  enumertate the states and extract the time derivatives and stored it is a dict
time_derivative_dict={}
def get_time_derivative(d):

  for key in d.keys():
    vi = []
    li = []
    temp_dic={}
    for state in d[key]['states'].keys():
      li.append(state)

      if 'time_derivative' in d[key]['states'][state].keys():

        vi.append(d[key]['states'][state]['time_derivative'])
      else:
        vi.append(None)
    for i in range(len(vi)):
      temp_dic[li[i]]=vi[i]

    time_derivative_dict[key] = temp_dic
get_time_derivative(dic)
print("time derivative dict>>>",time_derivative_dict)

for node,states in time_derivative_dict.items():
  for state in states.keys():
    if time_derivative_dict[node][state] is not None:

      _add_mdf_function(
        "evaluate_{}_{}_next_value".format(node,state),
        description="computing the next value of stateful parameter {}".format(state),
        arguments= arg_dict[node],
        expression_string= str(state)+"+" "(dt*"+str(time_derivative_dict[node][state])+")",
    )
    else:
      print('No need to create MDF function for node %s, state %s since there is no expression for time derivative!'%(node, state))

print(mdf_functions)
  # print(state,derivative)
#
# def check_Cond(self,**kwargs):
#
#     if 'dt' in kwargs and kwargs['time']==None:
#       raise Exception("Required parameter time is missing")
#     if 'time' in kwargs and kwargs['dt']==None:
#       raise Exception("Required parameter dt is missing")
#
#
#
#
# #second step 3 - computing the next value of the states/ stateful params and update the args_dict recursively computed
# #call a function and return the updated dictionary
#
#
# def update_params(expression, args_dict, stateful_parameter):
#    # rate =
#     #period =
#     #amp =
#     #dt =
#     #level =
#
#
#     args_dict[stateful_parameter] = args_dict[stateful_parameter] + eval(expression) * args_dict['dt'] # dl/dt
#
