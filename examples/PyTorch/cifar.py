# Code take from: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Turn training on or off
do_train = False

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# helper functions
#%%
def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(15, 6))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{}, {:.1f}%\n(label: {})".format(
                classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig


#%%


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():

    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # datasets
    trainset = torchvision.datasets.FashionMNIST(
        "examples/PyTorch/data", download=True, train=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        "examples/PyTorch/data", download=True, train=False, transform=transform
    )

    # dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    # constant for classes
    classes = (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    )

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter("examples/PyTorch/runs/fashion_mnist_experiment_1")

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # write to tensorboard
    # writer.add_image('four_fashion_mnist_images', img_grid)

    writer.add_graph(net, images)
    writer.close()

    running_loss = 0.0
    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:  # every 1000 mini-batches...

                # ...log the running loss
                writer.add_scalar(
                    "training loss", running_loss / 1000, epoch * len(trainloader) + i
                )

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                # writer.add_figure('predictions vs. actuals',
                #                   plot_classes_preds(net, inputs, labels),
                #                   global_step=epoch * len(trainloader) + i)
                running_loss = 0.0
    print("Finished Training")

    images, labels = dataiter.next()
    writer.add_figure(
        "predictions vs. actuals", plot_classes_preds(net, images, labels)
    )

    return net


if __name__ == "__main__":

    if do_train:
        net = train()
    else:
        net = Net()

    # Make up some test images
    images = torch.ones(size=[4, 1, 28, 28])

    # Turn on eval mode for model to get rid of any randomization due to things like BatchNorm or Dropout
    net.eval()

    # Run the model once to get some ground truth outpot (from PyTorch)
    output = net(images).detach().numpy()

    from modeci_mdf.mdf import Model
    from modeci_mdf.execution_engine import EvaluableGraph
    from modeci_mdf.interfaces.pytorch import pytorch_to_mdf

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=net,
        args=(images,),
        example_outputs=output,
        trace=True,
    )

    mdf_model.to_json_file("cifar.json")

    # Get the graph
    mdf_graph = mdf_model.graphs[0]

    #%%
    # Add inputs to the parameters dict so we can feed this to the EvaluableGraph for initialization of all
    # graph inputs.
    params_dict["input1"] = images.numpy()

    # Save the initializer so we can run this example as a standalone without training the model again
    if do_train:
        np.savez("cifar_weights.npz", **params_dict)

    # Evaluate the model via the MDF scheduler
    eg = EvaluableGraph(graph=mdf_graph, verbose=False)
    eg.evaluate(initializer=params_dict)
    mdf_output = eg.enodes["Gemm_23"].evaluable_outputs["_23"].curr_value

    assert np.allclose(output, mdf_output, atol=1e-5)

    # fig = plt.figure(figsize=(10, 5))
    # plt.imshow(output)
    # fig.axes[0].set_title('PyTorch Model Output', fontweight="bold", size=20)
    # writer.add_figure('PyTorch Model Output', fig)

    # fig, axes = plt.subplots(2,1,figsize=(10,10))
    # axes[0].imshow(output)
    # axes[0].set_title('PyTorch Model Output', fontweight="bold", size=20)
    # axes[1].imshow(mdf_output)
    # axes[1].set_title('MDF\\WebGME Model Output', fontweight="bold", size=20)

    # fig = plt.figure(figsize=(10, 5))
    # plt.imshow(mdf_output)
    # fig.axes[0].set_title('MDF Model Output', fontweight="bold", size=20)
