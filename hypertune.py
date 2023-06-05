import random
from src import datasets
from src.models import rnn_models, metrics, train_model
from src.settings import SearchSpace, SearchSpaceAttention, SearchSpaceTransformer, TrainerSettings, presets
from pathlib import Path
from ray.tune import JupyterNotebookReporter
from ray import tune
import torch
import ray
from typing import Any, Dict, List, Optional, Union, cast
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from loguru import logger
from filelock import FileLock
import mlflow
import argparse

def train(config: Dict, model_class, epochs, model_type, checkpoint_dir=None):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """

    # we lock the datadir to avoid parallel instances trying to
    # access the datadir
    with FileLock(presets.datadir / ".lock"):
        trainstreamer, teststreamer = datasets.get_arabic(
            presets
        )

    # we set up the metric
    accuracy = metrics.Accuracy()
    
    # and create the model with the config
    # model = rnn_models.GRUmodel(config)
    # model = rnn_models.LSTMmodel(config)

    model = model_class(config)
    
    trainersettings = TrainerSettings(
        epochs=epochs,
        metrics=[accuracy],
        logdir=presets.logdir,
        train_steps=len(trainstreamer),
        valid_steps=len(teststreamer),
        tunewriter=["ray", "mlflow"],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )
    # because we set tunewriter=["ray"]
    # the trainloop wont try to report back to tensorboard,
    # but will report back with tune.report
    # this way, ray will know whats going on,
    # and can start/pause/stop a loop.
    # This is why we set earlystop_kwargs=None, because we
    # are handing over this control to ray.

    if model_type == "Transformer":
        trainer = train_model.TransformerTrainer(
            model=model,
            settings=trainersettings,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            traindataloader=trainstreamer.stream(),
            validdataloader=teststreamer.stream(),
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        )
    else:
        trainer = train_model.Trainer(
            model=model,
            settings=trainersettings,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            traindataloader=trainstreamer.stream(),
            validdataloader=teststreamer.stream(),
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        )

    trainer.loop()

# Define argument parser
parser = argparse.ArgumentParser(description='Model Type for Hyperparameter Tuning')
parser.add_argument('--model_type', type=str, required=True, 
                    help='Type of model to use for hyperparameter tuning. Options: "GRU", "LSTM", "AttentionGRU", "Transformer"')
parser.add_argument('--epochs', type=int, required=False, default=50,
                    help='Number of epochs for training.')
args = parser.parse_args()

if __name__ == "__main__":
    print("aa")
    model_type = args.model_type
    epochs = args.epochs

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arabic_digits")

    with mlflow.start_run():     
        ray.init()

        mlflow.set_tag("dev", "linksmith")
        config : Union[SearchSpace, SearchSpaceAttention, SearchSpaceTransformer]
        
        if model_type == "GRU":
            config = SearchSpace(
                input_size=13,
                output_size=20, 
                hidden_size=tune.randint(16, 128),
                dropout=tune.uniform(0.0, 0.3),   
                num_layers=tune.randint(2, 5),     
                tune_dir=Path("models/ray").resolve(),
                data_dir=presets.datadir.resolve(),
                use_mean=tune.choice([True, False])
            )
            model_class = rnn_models.GRUmodel
            mlflow.set_tag("model", "GRUmodel")

        elif model_type == "LSTM":
            config = SearchSpace(
                input_size=13,
                output_size=20, 
                hidden_size=tune.randint(16, 128),
                dropout=tune.uniform(0.0, 0.3),   
                num_layers=tune.randint(2, 5),     
                tune_dir=Path("models/ray").resolve(),
                data_dir=presets.datadir.resolve(),
                use_mean=tune.choice([True, False])
            )
            model_class = rnn_models.LSTMmodel
            mlflow.set_tag("model", "LSTMmodel")

        elif model_type == "AttentionGRU":
            config = SearchSpaceAttention(
                input_size=13,
                output_size=20, 
                # size_and_heads=tune.choice(SearchSpaceAttention.get_size_and_heads(4, 120, 4)),
                size_and_heads='120_30',
                dropout=tune.uniform(0.0, 0.3),  
                dropout_attention=tune.uniform(0.0, 0.3),  
                num_layers=4,     
                tune_dir=Path("models/ray").resolve(),
                data_dir=presets.datadir.resolve(),
                use_mean=False
            )
            model_class = rnn_models.AttentionGRUAarabic
            mlflow.set_tag("model", "AttentionGRU")

        elif model_type == "Transformer":
            # config = SearchSpaceTransformer(
            #     input_size=13,
            #     output_size=20, 
            #     #hidden_size=13*20,#tune.randint(256, 1024),
            #     hidden_size=256, 
            #     num_heads=13,
            #     dropout=tune.uniform(0.0, 0.5),   
            #     num_layers=tune.randint(2, 8),     
            #     tune_dir=Path("models/ray").resolve(),
            #     data_dir=presets.datadir.resolve(),
            #     use_mean=False,
            #     num_transformer_layers = tune.randint(4, 8)
            # )
            config = SearchSpaceTransformer(
                input_size=13,
                output_size=20,
                hidden_size=20*13,  # Adjust the hidden size according to your needs
                num_heads=13,  # Adjust the number of heads according to your needs
                dropout=tune.uniform(0.0, 0.5), # Adjust the dropout rate according to your needs
                num_layers=4,  # Adjust the number of transformer layers according to your needs
                tune_dir=Path("models/ray").resolve(),
                data_dir=presets.datadir.resolve(),
                use_mean=False,
                num_transformer_layers=4  # Adjust the number of transformer layers according to your needs
            )
            model_class = rnn_models.TransformerAarabic
            mlflow.set_tag("model", "Transformer")

        else:
            raise ValueError(f"Invalid model_type {model_type}. Choose from 'GRU', 'LSTM', 'AttentionGRU'")

        # Log hyperparameters to MLflow
        mlflow.log_param("datadir", f"{presets.datadir.resolve()}")
        mlflow.log_param("batchsize", f"{presets.batchsize}")
        mlflow.log_params(config.__dict__)

        reporter = CLIReporter()
        reporter.add_metric_column("Accuracy")

        bohb_hyperband = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=200,
            reduction_factor=3,
            stop_last_trials=False,
        )

        bohb_search = TuneBOHB()

        analysis = tune.run(
            lambda config: train(config, model_class, epochs, model_type),
            config=config.dict(),
            metric="test_loss",
            mode="min",
            progress_reporter=reporter,
            local_dir=config.tune_dir.__str__(),
            num_samples=1,
            search_alg=bohb_search,
            scheduler=bohb_hyperband,
            verbose=1,
        )

        ray.shutdown()
