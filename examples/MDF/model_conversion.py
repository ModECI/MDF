# from modeci_mdf import *
# from modeci_mdf.utils import load_mdf,load_mdf_json
# from modeci_mdf.execution_engine import EvaluableGraph
# # from modeci_mdf.standard_functions import mdf_functions
# import sys
#
# import graphviz
#
# import numpy as np
#
# from modeci_mdf.standard_functions import mdf_functions
# from modeci_mdf.interfaces.graphviz.importer import *
#
# # # # print(list(mdf_functions.keys())[-10:])
# f=load_mdf('abc_conditions.json').graphs[0]
#
# eg = EvaluableGraph(f, verbose=False)
#
# mdf_to_graphviz(f, engine=engines["d"], view_on_render=True,level=3)

import psyneulink as pnl
import graphviz

A = pnl.TransferMechanism(function=pnl.Linear(slope=2.0, intercept=2.0), name='A')
B = pnl.TransferMechanism(function=pnl.Logistic, name='B')
C = pnl.TransferMechanism(function=pnl.Exponential, name='C')
D = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(rate=0.05), name='D')

comp2 = pnl.Composition(name='model_ABCD_level_1', pathways=[[A,B,D], [A,C,D]])

comp2.show_graph(output_fmt='pdf')

