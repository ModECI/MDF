from modeci_mdf.mdf import Model, Graph, Node, Parameter, OutputPort, InputPort
from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.utils import simple_connect
def main():
    mod = Model(id="TrigFunctionsChain")
    mod_graph = Graph(id="arc_functions")
    mod.graphs.append(mod_graph)

    # First node: arctan
    arctan_node = Node(id="arctan_node")
    arctan_node.parameters.append(Parameter(id="input_value", value=1.0))  # Example input
    arctan_node.parameters.append(Parameter(
        id="arctan_result",
        function="arctan",
        args={"variable0": "input_value", "scale": 0.4},
    ))
    arctan_node.output_ports.append(OutputPort(id="output", value="arctan_result"))
    mod_graph.nodes.append(arctan_node)

    # Second node: arccos, receiving input from the first node
    arccos_node = Node(id="arccos_node")
    arccos_node.input_ports.append(InputPort(id="input_from_arctan"))
    arccos_node.parameters.append(Parameter(
        id="arccos_result",
        function="arccos",
        args={"variable0": "input_from_arctan", "scale": 0.3},
    ))
    arccos_node.output_ports.append(OutputPort(id="output", value="arccos_result"))
    mod_graph.nodes.append(arccos_node)

    # Third node: arcsin, receiving input from the second node
    arcsin_node = Node(id="arcsin_node")
    arcsin_node.input_ports.append(InputPort(id="input_from_arccos"))
    arcsin_node.parameters.append(Parameter(
        id="arcsin_result",
        function="arcsin",
        args={"variable0": "input_from_arccos", "scale": 0.2},
    ))
    arcsin_node.output_ports.append(OutputPort(id="output", value="arcsin_result"))
    mod_graph.nodes.append(arcsin_node)

    simple_connect(arctan_node, arccos_node, mod_graph)
    simple_connect(arccos_node, arcsin_node, mod_graph)

    # Create EvaluableGraph and evaluate the model
    eg = EvaluableGraph(mod_graph, verbose=True)
    eg.evaluate()
    
    # Print output of each node
    print("arctan_node output:", eg.enodes['arctan_node'].evaluable_outputs['output'].curr_value)
    print("arccos_node output:", eg.enodes['arccos_node'].evaluable_outputs['output'].curr_value)
    print("arcsin_node output:", eg.enodes['arcsin_node'].evaluable_outputs['output'].curr_value)

    #visualize the model
    mod.to_graph_image(engine="dot", output_format="png")

if __name__ == "__main__":
    main()
