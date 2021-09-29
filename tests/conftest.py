import pytest

import torch
import torch.nn as nn

from modeci_mdf.mdf import Model, Graph, Node, Edge, InputPort, OutputPort, Parameter

#
# def pytest_exception_interact(node, call, report):
#     excinfo = call.excinfo
#     if "script" in node.funcargs:
#         excinfo.traceback = excinfo.traceback.cut(path=node.funcargs["script"])
#     report.longrepr = node.repr_failure(excinfo)


@pytest.fixture
def simple_model_mdf():
    """
    A simple MDF model with two nodes. Input node has an input port with no receiving edge but it is not used
    because the output port uses a parameter instead.
    """
    mod = Model(id="Simple")
    mod_graph = Graph(id="simple_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input_node")
    input_node.parameters.append(Parameter(id="input_level", value=0.5))
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    processing_node = Node(id="processing_node")
    mod_graph.nodes.append(processing_node)

    processing_node.parameters.append(Parameter(id="lin_slope", value=0.5))
    processing_node.parameters.append(Parameter(id="lin_intercept", value=0))
    processing_node.parameters.append(Parameter(id="log_gain", value=3))

    ip1 = InputPort(id="input_port1")
    processing_node.input_ports.append(ip1)

    f1 = Parameter(
        id="linear_1",
        function="linear",
        args={"variable0": ip1.id, "slope": "lin_slope", "intercept": "lin_intercept"},
    )
    f2 = Parameter(
        id="logistic_1",
        function="logistic",
        args={"variable0": f1.id, "gain": "log_gain", "bias": 0, "offset": 0},
    )
    processing_node.parameters.append(f1)
    processing_node.parameters.append(f2)
    processing_node.output_ports.append(OutputPort(id="output_1", value="logistic_1"))

    e1 = Edge(
        id="input_edge",
        parameters={"weight": 0.55},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=processing_node.id,
        receiver_port=ip1.id,
    )

    mod_graph.edges.append(e1)

    return mod


@pytest.fixture
def inception_model_pytorch():
    """The InceptionBlocks model the WebGME folks provided as a test case for deepforge."""

    class InceptionBlocks(nn.Module):
        def __init__(self):
            super().__init__()

            self.asymmetric_pad = nn.ZeroPad2d((0, 1, 0, 1))
            self.conv2d = nn.Conv2d(
                in_channels=5, out_channels=64, kernel_size=(5, 5), padding=2, bias=True
            )
            self.prelu = nn.PReLU(init=0.0)
            self.averagepooling2d = nn.AvgPool2d((2, 2), stride=2, padding=0)
            self.conv2d2 = nn.Conv2d(
                in_channels=64,
                out_channels=48,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu2 = nn.PReLU(init=0.0)
            self.conv2d3 = nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu3 = nn.PReLU(init=0.0)
            self.conv2d4 = nn.Conv2d(
                in_channels=64,
                out_channels=48,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu4 = nn.PReLU(init=0.0)
            self.averagepooling2d2 = nn.AvgPool2d((2, 2), stride=1)
            self.conv2d5 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu5 = nn.PReLU(init=0.0)
            self.conv2d6 = nn.Conv2d(
                in_channels=64,
                out_channels=48,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu6 = nn.PReLU(init=0.0)
            self.conv2d7 = nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu7 = nn.PReLU(init=0.0)
            self.conv2d8 = nn.Conv2d(
                in_channels=240,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.conv2d9 = nn.Conv2d(
                in_channels=240,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.conv2d10 = nn.Conv2d(
                in_channels=240,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu8 = nn.PReLU(init=0.0)
            self.conv2d11 = nn.Conv2d(
                in_channels=64,
                out_channels=92,
                kernel_size=(5, 5),
                padding=2,
                bias=True,
            )
            self.prelu9 = nn.PReLU(init=0.0)
            self.prelu10 = nn.PReLU(init=0.0)
            self.averagepooling2d3 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            self.conv2d12 = nn.Conv2d(
                in_channels=240,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu11 = nn.PReLU(init=0.0)
            self.conv2d13 = nn.Conv2d(
                in_channels=64,
                out_channels=92,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu12 = nn.PReLU(init=0.0)
            self.prelu13 = nn.PReLU(init=0.0)
            self.averagepooling2d4 = nn.AvgPool2d((2, 2), stride=2, padding=0)
            self.conv2d14 = nn.Conv2d(
                in_channels=340,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu14 = nn.PReLU(init=0.0)
            self.conv2d15 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(5, 5),
                padding=2,
                bias=True,
            )
            self.prelu15 = nn.PReLU(init=0.0)
            self.conv2d16 = nn.Conv2d(
                in_channels=340,
                out_channels=128,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu16 = nn.PReLU(init=0.0)
            self.conv2d17 = nn.Conv2d(
                in_channels=340,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu17 = nn.PReLU(init=0.0)
            self.averagepooling2d5 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            self.conv2d18 = nn.Conv2d(
                in_channels=340,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu18 = nn.PReLU(init=0.0)
            self.conv2d19 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu19 = nn.PReLU(init=0.0)
            self.conv2d20 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu20 = nn.PReLU(init=0.0)
            self.conv2d21 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu21 = nn.PReLU(init=0.0)
            self.conv2d22 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu22 = nn.PReLU(init=0.0)
            self.averagepooling2d6 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            self.conv2d23 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu23 = nn.PReLU(init=0.0)
            self.conv2d24 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(5, 5),
                padding=2,
                bias=True,
            )
            self.prelu24 = nn.PReLU(init=0.0)
            self.conv2d25 = nn.Conv2d(
                in_channels=476,
                out_channels=128,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu25 = nn.PReLU(init=0.0)
            self.averagepooling2d7 = nn.AvgPool2d((2, 2), stride=2, padding=0)
            self.conv2d26 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu26 = nn.PReLU(init=0.0)
            self.averagepooling2d8 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            self.conv2d27 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu27 = nn.PReLU(init=0.0)
            self.conv2d28 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu28 = nn.PReLU(init=0.0)
            self.conv2d29 = nn.Conv2d(
                in_channels=476,
                out_channels=128,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu29 = nn.PReLU(init=0.0)
            self.dense = nn.Linear(22273, 1096, bias=True)
            self.prelu30 = nn.PReLU(init=0.0)
            self.dense2 = nn.Linear(1096, 1096, bias=True)
            self.prelu31 = nn.PReLU(init=0.0)
            self.dense3 = nn.Linear(1096, 180, bias=True)

        def forward(self, galaxy_images_output, ebv_output):
            conv2d_output = self.conv2d(galaxy_images_output)
            prelu_output = self.prelu(conv2d_output)
            averagepooling2d_output = self.averagepooling2d(prelu_output)
            conv2d_output2 = self.conv2d2(averagepooling2d_output)
            prelu_output2 = self.prelu2(conv2d_output2)
            conv2d_output3 = self.conv2d3(prelu_output2)
            prelu_output3 = self.prelu3(conv2d_output3)
            conv2d_output4 = self.conv2d4(averagepooling2d_output)
            prelu_output4 = self.prelu4(conv2d_output4)
            prelu_output4 = self.asymmetric_pad(prelu_output4)
            averagepooling2d_output2 = self.averagepooling2d2(prelu_output4)
            conv2d_output5 = self.conv2d5(averagepooling2d_output)
            prelu_output5 = self.prelu5(conv2d_output5)
            conv2d_output6 = self.conv2d6(averagepooling2d_output)
            prelu_output6 = self.prelu6(conv2d_output6)
            conv2d_output7 = self.conv2d7(prelu_output6)
            prelu_output7 = self.prelu7(conv2d_output7)
            concatenate_output = torch.cat(
                (prelu_output5, prelu_output3, prelu_output7, averagepooling2d_output2),
                dim=1,
            )
            conv2d_output8 = self.conv2d8(concatenate_output)
            conv2d_output9 = self.conv2d9(concatenate_output)
            conv2d_output10 = self.conv2d10(concatenate_output)
            prelu_output8 = self.prelu8(conv2d_output10)
            conv2d_output11 = self.conv2d11(prelu_output8)
            prelu_output9 = self.prelu9(conv2d_output11)
            prelu_output10 = self.prelu10(conv2d_output8)
            prelu_output10 = self.asymmetric_pad(prelu_output10)
            averagepooling2d_output3 = self.averagepooling2d3(prelu_output10)
            conv2d_output12 = self.conv2d12(concatenate_output)
            prelu_output11 = self.prelu11(conv2d_output12)
            conv2d_output13 = self.conv2d13(prelu_output11)
            prelu_output12 = self.prelu12(conv2d_output13)
            prelu_output13 = self.prelu13(conv2d_output9)
            concatenate_output2 = torch.cat(
                (
                    prelu_output13,
                    prelu_output12,
                    prelu_output9,
                    averagepooling2d_output3,
                ),
                dim=1,
            )
            averagepooling2d_output4 = self.averagepooling2d4(concatenate_output2)
            conv2d_output14 = self.conv2d14(averagepooling2d_output4)
            prelu_output14 = self.prelu14(conv2d_output14)
            conv2d_output15 = self.conv2d15(prelu_output14)
            prelu_output15 = self.prelu15(conv2d_output15)
            conv2d_output16 = self.conv2d16(averagepooling2d_output4)
            prelu_output16 = self.prelu16(conv2d_output16)
            conv2d_output17 = self.conv2d17(averagepooling2d_output4)
            prelu_output17 = self.prelu17(conv2d_output17)
            prelu_output17 = self.asymmetric_pad(prelu_output17)
            averagepooling2d_output5 = self.averagepooling2d5(prelu_output17)
            conv2d_output18 = self.conv2d18(averagepooling2d_output4)
            prelu_output18 = self.prelu18(conv2d_output18)
            conv2d_output19 = self.conv2d19(prelu_output18)
            prelu_output19 = self.prelu19(conv2d_output19)
            concatenate_output3 = torch.cat(
                (
                    prelu_output16,
                    prelu_output19,
                    prelu_output15,
                    averagepooling2d_output5,
                ),
                dim=1,
            )
            conv2d_output20 = self.conv2d20(concatenate_output3)
            prelu_output20 = self.prelu20(conv2d_output20)
            conv2d_output21 = self.conv2d21(prelu_output20)
            prelu_output21 = self.prelu21(conv2d_output21)
            conv2d_output22 = self.conv2d22(concatenate_output3)
            prelu_output22 = self.prelu22(conv2d_output22)
            prelu_output22 = self.asymmetric_pad(prelu_output22)
            averagepooling2d_output6 = self.averagepooling2d6(prelu_output22)
            conv2d_output23 = self.conv2d23(concatenate_output3)
            prelu_output23 = self.prelu23(conv2d_output23)
            conv2d_output24 = self.conv2d24(prelu_output23)
            prelu_output24 = self.prelu24(conv2d_output24)
            conv2d_output25 = self.conv2d25(concatenate_output3)
            prelu_output25 = self.prelu25(conv2d_output25)
            concatenate_output4 = torch.cat(
                (
                    prelu_output25,
                    prelu_output21,
                    prelu_output24,
                    averagepooling2d_output6,
                ),
                dim=1,
            )
            averagepooling2d_output7 = self.averagepooling2d7(concatenate_output4)
            conv2d_output26 = self.conv2d26(averagepooling2d_output7)
            prelu_output26 = self.prelu26(conv2d_output26)
            prelu_output26 = self.asymmetric_pad(prelu_output26)
            averagepooling2d_output8 = self.averagepooling2d8(prelu_output26)
            conv2d_output27 = self.conv2d27(averagepooling2d_output7)
            prelu_output27 = self.prelu27(conv2d_output27)
            conv2d_output28 = self.conv2d28(prelu_output27)
            prelu_output28 = self.prelu28(conv2d_output28)
            conv2d_output29 = self.conv2d29(averagepooling2d_output7)
            prelu_output29 = self.prelu29(conv2d_output29)
            concatenate_output5 = torch.cat(
                (prelu_output29, prelu_output28, averagepooling2d_output8), dim=1
            )
            flatten_output = torch.flatten(concatenate_output5)
            concatenate_output6 = torch.cat((flatten_output, ebv_output), dim=0)
            dense_output = self.dense(concatenate_output6)
            prelu_output30 = self.prelu30(dense_output)
            dense_output2 = self.dense2(prelu_output30)
            prelu_output31 = self.prelu31(dense_output2)
            dense_output3 = self.dense3(prelu_output31)

            return dense_output3

    torch.manual_seed(0)
    model = InceptionBlocks()
    model.eval()

    return model
