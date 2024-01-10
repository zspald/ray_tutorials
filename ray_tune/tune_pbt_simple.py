import os
import tempfile

import torch
import torch.optim as optim

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test_func
from ray.tune.schedulers import PopulationBasedTraining
import matplotlib.pyplot as plt

def train_convnet(config):
    # create data loaders, model, and optimizers
    step = 1
    train_loader, test_loader = get_data_loaders()
    model = ConvNet()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.01),
        momentum=config.get('momentum', 0.9),
    )

    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_dict = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"))

        # restore model and optimizer states from checkpoint
        model.load_state_dict(ckpt_dict["model_state_dict"])
        optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
        # set parameters accoriding to config
        for param_group in optimizer.param_groups:
            if 'lr' in config:
                param_group["lr"] = config["lr"]
            if 'momentum' in config:
                param_group["momentum"] = config["momentum"]

        last_step = ckpt_dict["step"]
        step = last_step + 1

    while True:
        ray.tune.examples.mnist_pytorch.train_func(model, optimizer,
                                                   train_loader)
        acc = test_func(model, test_loader)
        metrics = {"mean_accuracy": acc, "lr": config["lr"]}

        # save checkpoint every `checkpoint_interval` steps
        if step % config['checkpoint_interval'] == 0:
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(tmpdir, "checkpoint.pt"),
                )
                train.report(metrics,
                             checkpoint=Checkpoint.from_directory(tmpdir))
        else:
            train.report(metrics)

        step += 1


def main():
    perturbation_interval = 5
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        metric="mean_accuracy",
        mode="max",
        hyperparam_mutations={
            # distribution for resampling
            "lr": tune.uniform(0.001, 1),
            # allow perturbations within this set of categorical values
            "momentum": [0.8, 0.9, 0.99],
        },
    )

    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    tuner = tune.Tuner(
        train_convnet,
        run_config=train.RunConfig(
            name="pbt_test",
            # stop when we've reached threshold accuracy, or a max training
            # iteration, whichever comes first
            stop={"mean_accuracy": 0.96, "training_iteration": 10},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_score_attribute="mean_accuracy",
                num_to_keep=4,
        ),
        storage_path='/tmp/ray_results',
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=4,
        ),
        param_space={
            "lr": tune.uniform(0.001, 1),
            "momentum": tune.uniform(0.001, 1),
            "checkpoint_interval": perturbation_interval,
        },
    )
    
    result_grid = tuner.fit()

    # Get the best trial result
    best_result = result_grid.get_best_result(metric="mean_accuracy",
                                              mode="max")

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
    df.plot("training_iteration", "mean_accuracy")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.show()

if __name__ == "__main__":
    main()