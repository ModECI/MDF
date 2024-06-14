from modeci_mdf.mdf import *
import os
from modeci_mdf.utils import simple_connect
import sys
from modeci_mdf.execution_engine import EvaluableGraph
import matplotlib.pyplot as plt

def run_simulation(mod_graph, duration=2, dt=0.001):
    eg = EvaluableGraph(mod_graph, verbose=False)
    t = 0
    times = []
    Vs_values = []
    V_values = []
    i_L_values = []
    i_R_values = []
    i_C_values = []

    while t <= duration:
        times.append(t)
        if t == 0:
            eg.evaluate()
        else:
            eg.evaluate(time_increment=dt)
        
        i_L_values.append(eg.enodes["V"].evaluable_outputs["i_L_out"].curr_value)
        i_R_values.append(eg.enodes["V"].evaluable_parameters["i_R"].curr_value)
        i_C_values.append(eg.enodes["V"].evaluable_parameters["i_C"].curr_value)
        Vs_values.append(eg.enodes["V"].evaluable_parameters["Vs"].curr_value)
        V = eg.enodes["V"].evaluable_parameters["V"].curr_value
        V_values.append(V)
        t += dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(times, i_L_values, label='Inductor Current (i_L)')
    ax1.plot(times, i_R_values, label='Resistor Current (i_R)')
    ax1.plot(times, i_C_values, label='Capacitor Current (i_C)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current (A)')
    ax1.set_title('Currents in RLC Circuit')
    ax1.legend()

    ax2.plot(times, V_values, label='Response Voltage (V)', color='orange')
    ax2.plot(times, Vs_values, label='Source Voltage (Vs)', color='green')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_title('Voltage in RLC Circuit')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("switched_rlc_plot.png")
    plt.show()

    return times, i_L_values, i_R_values, i_C_values, V_values

def main():
    parameters = {
        "R": 100,  # Resistance in Ohms
        "L": 1,  # Inductance in Henrys
        "C": 1e-3,  # Capacitance in Farads
        "Vb": 0.1  # Battery Voltage in Volts
    }

    mod = Model(id="SwitchedRLC_Circuit")
    mod_graph = Graph(id="SwitchedRLC_Circuit")
    mod.graphs.append(mod_graph)

    node = Node(id="V")

    vs = ParameterCondition(id="off", test="time<0.5", value="0")
    vb = ParameterCondition(id="on", test="time>=0.5", value=parameters["Vb"])
    voltage = Parameter(id="Vs")
    voltage.conditions.append(vs)
    voltage.conditions.append(vb)
    node.parameters.append(voltage)

    node.parameters.append(Parameter(id="R", value=parameters["R"]))
    node.parameters.append(Parameter(id="L", value=parameters["L"]))
    node.parameters.append(Parameter(id="C", value=parameters["C"]))

    node.parameters.append(Parameter(id="time", default_initial_value=0, time_derivative="1"))
    node.parameters.append(Parameter(id="V", default_initial_value=0, time_derivative="i_C /C"))
    node.parameters.append(Parameter(id="i_R", value="V / R"))
    node.parameters.append(Parameter(id="i_L", default_initial_value=0, time_derivative="(Vs - V)/L"))
    node.parameters.append(Parameter(id="i_C", value="i_L-i_R"))

    node.output_ports.append(OutputPort(id="V_out", value="V"))
    node.output_ports.append(OutputPort(id="i_L_out", value="i_L"))
    mod_graph.nodes.append(node)
    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    mod.to_graph_image(
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=3,
        filename_root="switched_rlc_circuit",
        only_warn_on_fail=(os.name == "nt"),
    )

    if "-run" in sys.argv:
         times, i_L_values, i_R_values, i_C_values, V_values = run_simulation(mod_graph)

if __name__ == "__main__":
    main()
