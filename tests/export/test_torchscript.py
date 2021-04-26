import torch
import torch.nn as nn

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

from modeci_mdf.export.torchscript.converter import torchscript_to_mdf


def _check_model(model):
    """A helper function to JIT compile a function or torch.nn.Module into Torchscript and convert to MDF and check it"""

    # JIT compile the model into TorchScript
    model = torch.jit.script(model)

    mdf_model = torchscript_to_mdf(model)

    # Generate JSON
    json_str = mdf_model.to_json()

    # Load the JSON
    # load_mdf_json()


def test_simple_module():
    """Test a simple torch.nn.Module"""

    class Simple(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    _check_model(Simple())


def test_simple_function():
    """Test a simple function"""

    def simple(x, y):
        return x + y

    _check_model(simple)


def test_inception():
    """Test an Inception model"""

    # pytorch does not have padding=same. To make shapes match,
    # add padding 1 randomly to left/right and top/bottom
    def custom_padding(input_tensor):
        batch, channels, x, y = input_tensor.shape
        custom = torch.zeros((batch, channels, x + 1, y + 1))

        # height_shift = torch.randint(2, (1,))
        # width_shift = torch.randint(2, (1,))
        height_shift = 0
        width_shift = 1

        height_indices = [idx + height_shift for idx in (0, x)]
        width_indices = [idx + width_shift for idx in (0, y)]

        custom[
            :,
            :,
            height_indices[0] : height_indices[1],
            width_indices[0] : width_indices[1],
        ] = input_tensor[:, :, :, :]

        return custom

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, galaxy_images_output, ebv_output):
            conv2d = nn.Conv2d(
                in_channels=5, out_channels=64, kernel_size=(5, 5), padding=2, bias=True
            )
            conv2d_output = conv2d(galaxy_images_output)

            prelu = nn.PReLU(init=0.0)
            prelu_output = prelu(conv2d_output)

            averagepooling2d = nn.AvgPool2d((2, 2), stride=2, padding=0)
            averagepooling2d_output = averagepooling2d(prelu_output)

            conv2d2 = nn.Conv2d(
                in_channels=64,
                out_channels=48,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output2 = conv2d2(averagepooling2d_output)

            prelu2 = nn.PReLU(init=0.0)
            prelu_output2 = prelu2(conv2d_output2)

            conv2d3 = nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            conv2d_output3 = conv2d3(prelu_output2)

            prelu3 = nn.PReLU(init=0.0)
            prelu_output3 = prelu3(conv2d_output3)

            conv2d4 = nn.Conv2d(
                in_channels=64,
                out_channels=48,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output4 = conv2d4(averagepooling2d_output)

            prelu4 = nn.PReLU(init=0.0)
            prelu_output4 = prelu4(conv2d_output4)

            prelu_output4 = custom_padding(prelu_output4)

            averagepooling2d2 = nn.AvgPool2d((2, 2), stride=1)
            averagepooling2d_output2 = averagepooling2d2(prelu_output4)

            conv2d5 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output5 = conv2d5(averagepooling2d_output)

            prelu5 = nn.PReLU(init=0.0)
            prelu_output5 = prelu5(conv2d_output5)

            conv2d6 = nn.Conv2d(
                in_channels=64,
                out_channels=48,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output6 = conv2d6(averagepooling2d_output)

            prelu6 = nn.PReLU(init=0.0)
            prelu_output6 = prelu6(conv2d_output6)

            conv2d7 = nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output7 = conv2d7(prelu_output6)

            prelu7 = nn.PReLU(init=0.0)
            prelu_output7 = prelu7(conv2d_output7)

            concatenate_output = torch.cat(
                (prelu_output5, prelu_output3, prelu_output7, averagepooling2d_output2),
                dim=1,
            )

            conv2d8 = nn.Conv2d(
                in_channels=240,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output8 = conv2d8(concatenate_output)

            conv2d9 = nn.Conv2d(
                in_channels=240,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output9 = conv2d9(concatenate_output)

            conv2d10 = nn.Conv2d(
                in_channels=240,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output10 = conv2d10(concatenate_output)

            prelu8 = nn.PReLU(init=0.0)
            prelu_output8 = prelu8(conv2d_output10)

            conv2d11 = nn.Conv2d(
                in_channels=64,
                out_channels=92,
                kernel_size=(5, 5),
                padding=2,
                bias=True,
            )
            conv2d_output11 = conv2d11(prelu_output8)

            prelu9 = nn.PReLU(init=0.0)
            prelu_output9 = prelu9(conv2d_output11)

            prelu10 = nn.PReLU(init=0.0)
            prelu_output10 = prelu10(conv2d_output8)

            prelu_output10 = custom_padding(prelu_output10)

            averagepooling2d3 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            averagepooling2d_output3 = averagepooling2d3(prelu_output10)

            conv2d12 = nn.Conv2d(
                in_channels=240,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output12 = conv2d12(concatenate_output)

            prelu11 = nn.PReLU(init=0.0)
            prelu_output11 = prelu11(conv2d_output12)

            conv2d13 = nn.Conv2d(
                in_channels=64,
                out_channels=92,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            conv2d_output13 = conv2d13(prelu_output11)

            prelu12 = nn.PReLU(init=0.0)
            prelu_output12 = prelu12(conv2d_output13)

            prelu13 = nn.PReLU(init=0.0)
            prelu_output13 = prelu13(conv2d_output9)

            concatenate_output2 = torch.cat(
                (
                    prelu_output13,
                    prelu_output12,
                    prelu_output9,
                    averagepooling2d_output3,
                ),
                dim=1,
            )

            averagepooling2d4 = nn.AvgPool2d((2, 2), stride=2, padding=0)
            averagepooling2d_output4 = averagepooling2d4(concatenate_output2)

            conv2d14 = nn.Conv2d(
                in_channels=340,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output14 = conv2d14(averagepooling2d_output4)

            prelu14 = nn.PReLU(init=0.0)
            prelu_output14 = prelu14(conv2d_output14)

            conv2d15 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(5, 5),
                padding=2,
                bias=True,
            )
            conv2d_output15 = conv2d15(prelu_output14)

            prelu15 = nn.PReLU(init=0.0)
            prelu_output15 = prelu15(conv2d_output15)

            conv2d16 = nn.Conv2d(
                in_channels=340,
                out_channels=128,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output16 = conv2d16(averagepooling2d_output4)

            prelu16 = nn.PReLU(init=0.0)
            prelu_output16 = prelu16(conv2d_output16)

            conv2d17 = nn.Conv2d(
                in_channels=340,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output17 = conv2d17(averagepooling2d_output4)

            prelu17 = nn.PReLU(init=0.0)
            prelu_output17 = prelu17(conv2d_output17)

            prelu_output17 = custom_padding(prelu_output17)

            averagepooling2d5 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            averagepooling2d_output5 = averagepooling2d5(prelu_output17)

            conv2d18 = nn.Conv2d(
                in_channels=340,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output18 = conv2d18(averagepooling2d_output4)

            prelu18 = nn.PReLU(init=0.0)
            prelu_output18 = prelu18(conv2d_output18)

            conv2d19 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            conv2d_output19 = conv2d19(prelu_output18)

            prelu19 = nn.PReLU(init=0.0)
            prelu_output19 = prelu19(conv2d_output19)

            concatenate_output3 = torch.cat(
                (
                    prelu_output16,
                    prelu_output19,
                    prelu_output15,
                    averagepooling2d_output5,
                ),
                dim=1,
            )

            conv2d20 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output20 = conv2d20(concatenate_output3)

            prelu20 = nn.PReLU(init=0.0)
            prelu_output20 = prelu20(conv2d_output20)

            conv2d21 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            conv2d_output21 = conv2d21(prelu_output20)

            prelu21 = nn.PReLU(init=0.0)
            prelu_output21 = prelu21(conv2d_output21)

            conv2d22 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output22 = conv2d22(concatenate_output3)

            prelu22 = nn.PReLU(init=0.0)
            prelu_output22 = prelu22(conv2d_output22)

            prelu_output22 = custom_padding(prelu_output22)

            averagepooling2d6 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            averagepooling2d_output6 = averagepooling2d6(prelu_output22)

            conv2d23 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output23 = conv2d23(concatenate_output3)

            prelu23 = nn.PReLU(init=0.0)
            prelu_output23 = prelu23(conv2d_output23)

            conv2d24 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(5, 5),
                padding=2,
                bias=True,
            )
            conv2d_output24 = conv2d24(prelu_output23)

            prelu24 = nn.PReLU(init=0.0)
            prelu_output24 = prelu24(conv2d_output24)

            conv2d25 = nn.Conv2d(
                in_channels=476,
                out_channels=128,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output25 = conv2d25(concatenate_output3)

            prelu25 = nn.PReLU(init=0.0)
            prelu_output25 = prelu25(conv2d_output25)

            concatenate_output4 = torch.cat(
                (
                    prelu_output25,
                    prelu_output21,
                    prelu_output24,
                    averagepooling2d_output6,
                ),
                dim=1,
            )

            averagepooling2d7 = nn.AvgPool2d((2, 2), stride=2, padding=0)
            averagepooling2d_output7 = averagepooling2d7(concatenate_output4)

            conv2d26 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output26 = conv2d26(averagepooling2d_output7)

            prelu26 = nn.PReLU(init=0.0)
            prelu_output26 = prelu26(conv2d_output26)

            prelu_output26 = custom_padding(prelu_output26)

            averagepooling2d8 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            averagepooling2d_output8 = averagepooling2d8(prelu_output26)

            conv2d27 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output27 = conv2d27(averagepooling2d_output7)

            prelu27 = nn.PReLU(init=0.0)
            prelu_output27 = prelu27(conv2d_output27)

            conv2d28 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            conv2d_output28 = conv2d28(prelu_output27)

            prelu28 = nn.PReLU(init=0.0)
            prelu_output28 = prelu28(conv2d_output28)

            conv2d29 = nn.Conv2d(
                in_channels=476,
                out_channels=128,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            conv2d_output29 = conv2d29(averagepooling2d_output7)

            prelu29 = nn.PReLU(init=0.0)
            prelu_output29 = prelu29(conv2d_output29)

            concatenate_output5 = torch.cat(
                (prelu_output29, prelu_output28, averagepooling2d_output8), dim=1
            )

            flatten_output = torch.flatten(concatenate_output5)

            concatenate_output6 = torch.cat((flatten_output, ebv_output), dim=0)

            dense = nn.Linear(22273, 1096, bias=True)
            dense_output = dense(concatenate_output6)

            prelu30 = nn.PReLU(init=0.0)
            prelu_output30 = prelu30(dense_output)

            dense2 = nn.Linear(1096, 1096, bias=True)
            dense_output2 = dense2(prelu_output30)

            prelu31 = nn.PReLU(init=0.0)
            prelu_output31 = prelu31(dense_output2)

            dense3 = nn.Linear(1096, 180, bias=True)
            dense_output3 = dense3(prelu_output31)

            return dense_output3

    galaxy_images_output = torch.zeros((1, 5, 64, 64))
    ebv_output = torch.zeros((1,))

    model = Model()

    # Force things to the cpu
    model.to(torch.device("cpu"))

    # Turn off any random stuff, DropOut, BatchNorm
    model.eval()

    torch.manual_seed(0)
    expected = model(galaxy_images_output, ebv_output)

    torch.manual_seed(0)
    jit_model = torch.jit.trace(model, (galaxy_images_output, ebv_output))

    mdf_model = torchscript_to_mdf(jit_model)
    yaml_str = mdf_model.to_yaml()
