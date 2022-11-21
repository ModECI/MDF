from modeci_mdf.mdf import *
import numpy as np


def create_rnn_node(id, N, g, seed=1234):

    np.random.seed(seed)

    ## RNN node...
    rnn_node = Node(id=id)
    ext_ip = InputPort(id="ext_input")
    rnn_node.input_ports.append(ext_ip)

    fb_ip = InputPort(id="fb_input", default_value=0)
    rnn_node.input_ports.append(fb_ip)

    default_initial_value = np.zeros(N)
    default_initial_value = 2 * np.random.random(N) - 1

    M = Parameter(id="M", value=2 * np.random.random((N, N)) - 1)
    rnn_node.parameters.append(M)

    g = Parameter(id="g", value=g)
    rnn_node.parameters.append(g)

    x = Parameter(
        id="x",
        default_initial_value=default_initial_value,
        time_derivative=f"-x + g*int_fb + {ext_ip.id} + {fb_ip.id}",
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
