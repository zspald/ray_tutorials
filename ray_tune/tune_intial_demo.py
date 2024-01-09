import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ray import train, tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

import os


N_EPOCHS = 10
EPOCH_SIZE = 512
TEST_SIZE = 256
# define run configuration ifno
RUN_CONFIG = RunConfig(
    storage_path = os.path.abspath("run_results"),
    name="train_ddp_fmnist_linear"
)



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.fc = nn.Linear(in_features=3 * 8 * 8, out_features=10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 3 * 8 * 8)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    

def train_func(model, optimizer, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # set this just for the example to run quickly
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test_func(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # set this just for the example to run quickly
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


def train_mnist(config):
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = DataLoader(
        datasets.MNIST("~/data",
                       train=True,
                       download=True,
                       transform=mnist_transforms,
                       ),
        batch_size=64,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("~/data",
                       train=False,
                       transform=mnist_transforms,
                       ),
        batch_size=64,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet().to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
    )

    for e in range(N_EPOCHS):
        train_func(model, optimizer, train_loader)
        acc = test_func(model, test_loader)

        # send the current training result back to Tune
        train.report({'mean_accuracy': acc})

        if e % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")

search_space = {
    'lr': tune.loguniform(1e-4, 1e-1),
    'momentum': tune.uniform(0.1, 0.9)
}
hp_search = HyperOptSearch(metric="mean_accuracy", mode="max")

tune_config = tune.TuneConfig(
    num_samples=20,
    # scheduler=ASHAScheduler(
    #     metric="mean_accuracy",
    #     mode="max",
    # ),
    search_alg=hp_search,
)


if __name__ == "__main__":
    # uncomment this to enable distributed execution
    # ray.init(address="auto")

    # download dataset first
    datasets.MNIST("~/data", train=True, download=True)

    tuner = tune.Tuner(
        train_mnist,
        tune_config=tune_config,
        param_space=search_space,
        run_config=RUN_CONFIG,
    )
    results = tuner.fit()
    # print(results)
    dfs = {result.path: result.metrics_dataframe for result in results}

    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)
    plt.show()