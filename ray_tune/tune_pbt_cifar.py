import argparse
import os
import tempfile

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from filelock import FileLock
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm
import matplotlib.pyplot as plt

import ray
from ray import train, tune
from ray.train import Checkpoint, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner


def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) // train.get_context().get_world_size()
    model.train()  # training mode for layers like dropout, etc.
    # for batch, (X, y) in enumerate(dataloader):
    for X, y in tqdm(dataloader, desc="Train epoch"):
        # prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset) // train.get_context().get_world_size()
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches  # avg loss for epoch
    correct /= size
    print(
        f"Test error: \n "
        f" Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return {'loss': test_loss}


def update_optimizer_config(optimizer, config):
    for param_group in optimizer.param_groups:
        for param, val in config.items():
            param_group[param] = val


def train_func(config):
    epochs = config.get('epochs', 3)

    model = resnet18()

    # prepare model needs to be called before setting optimizer
    if not train.get_checkpoint():
        model = train.torch.prepare_model(model)


    # create optimizer
    optimizer_config = {
        'lr': config.get('lr'),
        'momentum': config.get('momentum'),
    }
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_config)

    starting_epoch = 0
    if train.get_checkpoint():
        with train.get_checkpoint().as_directory() as ckpt_dir:
            ckpt_dict = torch.load(os.path.join(ckpt_dir, 'ckpt.pt'))
        
        # load model
        model = train.torch.prepare_model(model)
        model_state = ckpt_dict['model']
        model.load_state_dict(model_state)

        # load optimizer
        optimizer_state = ckpt_dict['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state)

        # optimizers configs are being mutated by PBT and passed in through
        # config, so we need to updatet the optimizer loaded from the ckpt
        update_optimizer_config(optimizer, optimizer_config)

        # current epoch increments the loaded epoch by 1
        ckpt_epoch = ckpt_dict['epoch']
        starting_epoch = ckpt_epoch + 1

    # load in training and validation data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # data augmentation?
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])  # mean/std transformation for CIFAR10 w/ data augmentation
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])  # mean/std transformation for CIFAR10

    data_dir = config.get('data_dir', os.path.expanduser('~/data'))
    os.makedirs(data_dir, exist_ok=True)
    with FileLock(os.path.join(data_dir, '.ray.lock')):
        train_dataset = CIFAR10(data_dir, train=True, download=True,
                                transform=transform_train)
        val_dataset = CIFAR10(data_dir, train=False, download=True,
                               transform=transform_test)

    if config.get('test_mode'):
        train_dataset = Subset(train_dataset, list(range(64)))
        val_dataset = Subset(val_dataset, list(range(64)))

    worker_batch_size = config['batch_size'] // train.get_context().get_world_size()
    train_loader = DataLoader(train_dataset, batch_size=worker_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=worker_batch_size)

    # make sure data is on correct device
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)

    # create loss
    criterion = nn.CrossEntropyLoss()

    for epoch in range(starting_epoch, epochs): # generalized for ckpt
        train_epoch(train_loader, model, criterion, optimizer)
        result = validate_epoch(val_loader, model, criterion)

        with tempfile.TemporaryDirectory() as tmp_dir:
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                },
                os.path.join(tmp_dir, 'ckpt.pt'),
            )
            train.report(result, checkpoint=Checkpoint.from_directory(tmp_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', '-n', type=int, default=2,
                       help='Number of workers to use for training.')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train for.')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Finish quickly for testing.')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for training.')
    parser.add_argument('--data-dir', type=str, default='~/data',
                        required=False, help='Directory for downloading data.')
    parser.add_argument('--synch', action='store_true',
                        help='Use synchronous PBT.')
    args, _ = parser.parse_known_args()
    
    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
        ),
    )
    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=args.num_epochs//10,
        burn_in_period=2*(args.num_epochs//10),
        hyperparam_mutations={
            'train_loop_config':{
                # distribution for resampling
                'lr': tune.loguniform(1e-3, 1e-1),
                # allow perturbations within this set of categorical values
                'momentum': [0.5, 0.8, 0.9, 0.99],
            }
        },
        synch=args.synch,
    )

    tuner = Tuner(
        trainer,
        param_space={
            'train_loop_config': {
                'lr': tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]),
                'momentum': 0.8,
                'batch_size': 512 * args.num_workers,
                'test_mode': args.smoke_test,  # whether to subset data
                'data_dir': args.data_dir,
                'epochs': args.num_epochs,
            }
        },
        tune_config=TuneConfig(
            num_samples=4,
            metric='loss',
            mode='min',
            scheduler=pbt_scheduler,
        ),
        run_config=RunConfig(
            storage_path=(os.path.expanduser('~') + '/workspace/'),
            stop={'training_iteration': 3 if args.smoke_test else args.num_epochs},
            failure_config=FailureConfig(max_failures=3),  # fault tolerance
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric='loss', mode='min')
    print(f"Results: {best_result}")

    # Print `path` where checkpoints are stored
    print('Best result path:', best_result.path)

    # Print the best trial `config` reported at the last iteration
    # NOTE: This config is just what the trial ended up with at the last iteration.
    # See the next section for replaying the entire history of configs.
    print("Best final iteration hyperparameter config:\n", best_result.config)

    # Plot the learning curve for the best trial
    df = best_result.metrics_dataframe
    # Deduplicate, since PBT might introduce duplicate data
    df = df.drop_duplicates(subset="training_iteration", keep="last")
    df.plot("training_iteration", "loss")
    plt.xlabel("Training Iterations")
    plt.ylabel("Validation Loss")
    plt.show()