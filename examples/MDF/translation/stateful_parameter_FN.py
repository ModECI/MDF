from modeci_mdf.mdf import *
import sys

def main():
	mod = Model(id="New_Stateful_Parameters_FN")
	mod_graph = Graph(id="stateful_parameter_FN")
	mod.graphs.append(mod_graph)

   

	
	fn_node = Node(id="FNpop_0", parameters={"initial_w": 0.,
						"initial_v": -1.,
						"a_v": -0.3333333333333333,
						"b_v": 0,
						"c_v": 1,
						"d_v": 1,
						"e_v": -1,
						"f_v": 1,
						"time_constant_v": 1,
						"a_w": 1,
						"b_w": -0.8,
						"c_w": 0.7,
						"time_constant_w": 12.5,
						"threshold": -1,
						"mode": 1,
						"uncorrelated_activity": 0,
						"Iext": 0,
						"MSEC": 0.001,
						"dt": 0.00005})
	

	
	v = Stateful_Parameter(id="V", default_initial_value= fn_node.parameters['initial_v'], value="update_V")
	fn_node.stateful_parameters.append(v)

	w = Stateful_Parameter(id="W", default_initial_value=fn_node.parameters['initial_w'], value="update_W")
	fn_node.stateful_parameters.append(w)

	fv = Function(
		id="update_V",
		function="linear",
		args={"variable0": "time_derivative_V", "slope":"dt", "intercept": v.id},
	)

	fw = Function(
		id="update_W",
		function="linear",
		args={"variable0": "time_derivative_W", "slope": "dt", "intercept":w.id },
	)

	fdv_dt = Function(
		id="time_derivative_V",
		function="time_derivative_FN_V",
		args={"variable0":v.id , "variable1":w.id ,"a_v":"a_v", "threshold":"threshold", "b_v":"b_v", "c_v":"c_v", "d_v":"d_v", "e_v":"e_v", "f_v":"f_v","Iext":"Iext", "time_constant_v":"time_constant_v", "MSEC":"MSEC"},
	)

	


	fdw_dt = Function(
		id="time_derivative_W",
		function="time_derivative_FN_W",
		args={"variable0":v.id,"variable1":w.id, "a_w":"a_w", "b_w":"b_w", "c_w":"c_w", "mode":"mode","uncorrelated_activity":"uncorrelated_activity", "time_constant_w":"time_constant_w", "MSEC":"MSEC"},
	)


	
	fn_node.functions.append(fv)
	fn_node.functions.append(fw)

	
	fn_node.functions.append(fdv_dt)
	fn_node.functions.append(fdw_dt)

	

	op = OutputPort(id="OUTPUT", value="V")
	fn_node.output_ports.append(op)

	mod_graph.nodes.append(fn_node)

	new_file = mod.to_json_file("%s.json" % mod.id)
	new_file = mod.to_yaml_file("%s.yaml" % mod.id)
	if "-run" in sys.argv:
		verbose = True
		from modeci_mdf.utils import load_mdf, print_summary

		from modeci_mdf.scheduler import EvaluableGraph

		eg = EvaluableGraph(mod_graph, verbose)
		dt = 0.00005
		duration= 0.1
		#duration= 2
		t = 0

		times = []
		vv = []
		ww = []
		ww1 =[]
		while t<duration+0.00005:
			times.append(t)
			print("======   Evaluating at t = %s  ======"%(t))
			vv.append(float(eg.enodes['FNpop_0'].evaluable_stateful_parameters['V'].curr_value))
			ww.append(float(eg.enodes['FNpop_0'].evaluable_stateful_parameters['W'].curr_value))
			
			
			eg.evaluate() 
			# ww.append(eg.enodes['FNpop_0'].evaluable_outputs['OUTPUT'].curr_value)
			
			# for i in range(len(eg.enodes['FNpop_0'].evaluable_stateful_parameters['V'].curr_value)):
			# 	if not i in vv:
			# 		vv[i]=[]
			# 		ww[i]=[]
			# 	v = eg.enodes['FNpop_0'].evaluable_stateful_parameters['V'].curr_value[i]
			# 	w = eg.enodes['FNpop_0'].evaluable_stateful_parameters['W'].curr_value[i]
			# 	vv[i].append(v)
			# 	ww[i].append(w)
			t+=dt
			# times.append(t)
		print(ww[:5],vv[:5],times[:5])
		import matplotlib.pyplot as plt
		# for vi in vv:
		# 	plt.plot(times,vv[vi],label='V_%s'%vi)
		# 	plt.plot(times,ww[vi],label='W_%s'%vi)
		plt.plot(times,vv,label='V')
		plt.plot(times,ww,label='W')
		plt.legend()
		plt.show()
		plt.savefig('FN_stateful_vw_plot.jpg')
	return mod_graph

if __name__ == "__main__":
	main()
