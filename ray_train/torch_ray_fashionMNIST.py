import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm
from filelock import FileLock
import ray.train as train
from ray.train import ScalingConfig, RunConfig, Result
from ray.train.torch import TorchTrainer

import os
from typing import Dict
import tempfile

# define run configuration ifno
run_config = RunConfig(
    storage_path = os.path.abspath("ray_results"),
    name="train_ddp_fmnist_linear"
)


# model definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def get_dataloaders(batch_size):
    # transform to normalize input images
    # (compose puts multiple transforms together)
    transform = transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    # file lock to prevent data corruption with access by multiple processes
    with FileLock(os.path.expanduser("~/data.lock")):
        # dowload train dataset
        training_data = datasets.FashionMNIST(
            root="~/data",
            train=True,
            download=True,
            transform=transform,
        )

        # download test dataset
        test_data = datasets.FashionMNIST(
            root="~/data",
            train=False,
            download=True,
            transform=transform,
        )

    # create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader

def train_epoch(model, optimizer, loss_fn, dataloader):
    model.train()
    for X, y in tqdm(dataloader, desc="Train epoch"):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_epoch(model, loss_fn, dataloader):
    model.eval()
    test_loss, n_correct, n_total = 0, 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Test epoch"):
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()
            n_correct += (pred.argmax(1) == y).sum().item()
            n_total += y.shape[0]

    test_loss /= len(dataloader)
    acc = n_correct / n_total
    return test_loss, acc


def train_func_per_worker(config: Dict):
    lr = config['lr']
    batch_size = config['batch_size']
    epochs = config['epochs']

    # get data loaders inside worker training function
    train_dataloader, test_dataloader = get_dataloaders(batch_size)

    # [1] Prepare dataloader for distributed training
    # Shared the datasets among workers and move batches to the correct device
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)

    model = NeuralNetwork()

    # [2] Prepare and wrap model with DistribtedDataParallel
    # move to correct device
    model = train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # training loop
    for e in range(epochs):
        train_epoch(model, optimizer, loss_fn, train_dataloader)
        test_loss, acc = test_epoch(model, loss_fn, test_dataloader)

        # # [3] Report metrics to Ray Train & save model
        metrics = {'loss': test_loss, 'accuracy': acc}
        # best practice is to upload to temp dir before report to persistent
        # storage
        with tempfile.TemporaryDirectory() as tmp_dir:
            if train.get_context().get_world_rank() == 0:
                state_dict = dict(epoch=e, state_dict=model.state_dict())
                torch.save(state_dict, os.path.join(tmp_dir, "checkpoint.bin"))
                checkpoint = train.Checkpoint.from_directory(tmp_dir)
            else:
                checkpoint = None
            train.report(metrics, checkpoint=checkpoint)
        print("Done!")


def train_fashion_mnist(num_workers=2, use_gpu=False):
    global_batch_size = 512

    train_config = {
        'lr': 1e-3,
        'batch_size': global_batch_size // num_workers,
        'epochs': 10,
    }

    # configure computation resources
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
    )

    # initialize trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # [4] Start distributed training
    # run 'train_func_per_worker' on each worker
    result = trainer.fit()
    print(f'Training result: {result}')


def load_model_from_checkpoint(trial_path):
    # path to results from run config defined above with specified trial
    path = os.path.join(run_config.storage_path, run_config.name, trial_path)
    restored_result = Result.from_path(path)
    state_dict = torch.load(restored_result.checkpoint.value)
    print(type(state_dict))
    # print(f"Restored result: {restored_result}")



if __name__ == '__main__':
    # train_fashion_mnist(num_workers=4, use_gpu=False)
    trial_path = "TorchTrainer_6f9f5_00000_0_2024-01-08_11-53-23"
    load_model_from_checkpoint(trial_path)
    