import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # no activation on output
        return x


def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
        )
    ])

    # file lock because multiple workers will be downloading data and 
    # dataloader is not threadsafe
    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)
        
        return trainset, testset
    
def load_test_data():
    # load fake data for running a quick smoke-test
    trainset = torchvision.datasets.FakeData(
        128, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.FakeData(
        16, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    return trainset, testset


def train_cifar(config):
    net = Net(config["l1"], config["l2"])

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # load existing checkpoint through `get_checkpoint()` API
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as ckpt_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(ckpt_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    if config['smoke_test']:
        trainset, _ = load_test_data()
    else:
        trainset, _ = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        num_workers=0 if config['smoke_test'] else 2,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        num_workers=0 if config['smoke_test'] else 2,
    )

    # training 
    for e in range(config['epochs']):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print stats
            running_loss += loss.item()
            epoch_steps += 1
            if e % 2000 == 1999:
                print(f"[{e + 1}, {i + 1}] loss: {running_loss / epoch_steps}")
                running_loss = 0.0

    # validation
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_steps += 1

    # save checkpoint. this is automatically registered with Ray Tune and will
    # potentially be accessed through `get_checkpoint()` in future iterations.
    with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
        path = os.path.join(tmp_ckpt_dir, "checkpoint.pt")
        torch.save((net.state_dict(), optimizer.state_dict()), path)
        checkpoint = Checkpoint.from_directory(tmp_ckpt_dir)
        train.report(
            {'loss': val_loss / val_steps,
             'accuracy': correct / total},
            checkpoint=checkpoint,
        )
    print('Finished Training')

def test_best_model(best_result, smoke_test=False):
    best_model = Net(best_result.config['l1'], best_result.config['l2'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model.to(device)

    ckpt_path = os.path.join(best_result.checkpoint.to_directory(),
                             'checkpoint.pt')

    model_state, optimizer_state = torch.load(ckpt_path)
    best_model.load_state_dict(model_state)

    if smoke_test:
        _, testset = load_test_data()
    else:
        _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Best trial test set accuracy {correct / total}')

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1,
         smoke_test=False):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([128, 256, 512]),
        "epochs": tune.choice([5, 10, 20]),
        "smoke_test": smoke_test,
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_cifar),
            resources={"cpu": 2, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print(f'Best trial config: {best_result.config}')
    print(f'Best trial final validation loss: {best_result.metrics["loss"]}')
    print('Best trial final validation accuracy: '
          f'{best_result.metrics["accuracy"]}')
    
    test_best_model(best_result, smoke_test=smoke_test)

if __name__ == '__main__':
    main(num_samples=10, max_num_epochs=20, gpus_per_trial=1, smoke_test=False)
    
