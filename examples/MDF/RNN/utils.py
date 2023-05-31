from modeci_mdf.mdf import *
import numpy as np


def create_rnn_node(id, N, g, seed=1234):

    np.random.seed(seed)

    ## RNN node...
    rnn_node = Node(id=id)
    ipr1 = InputPort(id="ext_input", shape="(%i,)" % N)
    rnn_node.input_ports.append(ipr1)

    ipr2 = InputPort(id="fb_input", shape="(%i,)" % N)
    rnn_node.input_ports.append(ipr2)

    default_initial_value = np.zeros(N)
    default_initial_value = 2 * np.random.random(N) - 1

    M = Parameter(id="M", value=2 * np.random.random((N, N)) - 1)
    rnn_node.parameters.append(M)

    g = Parameter(id="g", value=g)
    rnn_node.parameters.append(g)

    x = Parameter(
        id="x",
        default_initial_value=default_initial_value,
        time_derivative="-x + g*int_fb + %s" % ipr1.id,
    )
    rnn_node.parameters.append(x)

    r = Parameter(id="r", function="tanh", args={"variable0": x.id, "scale": 1})
    # r = Parameter(id="r", value="x")
    rnn_node.parameters.append(r)

    int_fb = Parameter(id="int_fb", function="MatMul", args={"A": "M", "B": "r"})
    rnn_node.parameters.append(int_fb)

    op_x = OutputPort(id="out_port_x", value="x")
    rnn_node.output_ports.append(op_x)

    op_r = OutputPort(id="out_port_r", value="r")
    rnn_node.output_ports.append(op_r)

    return rnn_node
