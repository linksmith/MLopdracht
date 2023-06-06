import random
from src import datasets
from src.models import model as m, metrics, train_model
from src.settings import SearchSpace, SearchSpaceAttention, SearchSpaceTransformer, SearchSpaceGRUTransformer, TrainerSettings, presets
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
    model = model_class(config)
    
    trainersettings = TrainerSettings(
        epochs=epochs,
        metrics=[accuracy],
        logdir=presets.logdir,
        train_steps=len(trainstreamer),
        valid_steps=len(teststreamer),
        tunewriter=["ray", "mlflow", "tensorboard"],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )

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


def get_config_and_model_class(model_type):
    if model_type == "LSTM":
        config = get_lstm_search_space()
        model_class = m.LSTM

    elif model_type == "GRU":
        config = get_gru_search_space()
        model_class = m.GRU

    elif model_type == "GRUAttention":
        config = get_attention_search_space()
        model_class = m.GRUAttention

    elif model_type == "Transformer":
        config = get_transformer_search_space()
        model_class = m.Transformer

    elif model_type == "GRUTransformer":
        config = get_gru_transformer_search_space()
        model_class = m.GRUTransformer

    else:
        raise ValueError(f"Invalid model_type {model_type}. Choose from 'GRU', 'LSTM', 'GRUAttention, 'GRUTransformer'")
    
    return config, model_class


def get_lstm_search_space():
    return SearchSpace(
        input_size=13,
        output_size=20, 
        hidden_size=tune.randint(16, 128),
        dropout=tune.uniform(0.0, 0.3),   
        num_layers=tune.randint(2, 5),     
        tune_dir=Path("models/ray2").resolve(),
        data_dir=presets.datadir.resolve(),
        use_mean=tune.choice([True, False])
    )

# Define model config space
def get_gru_search_space():
    return SearchSpace(
        input_size=13,
        output_size=20, 
        hidden_size=tune.randint(16, 128),
        dropout=tune.uniform(0.0, 0.3),   
        num_layers=tune.randint(2, 5),     
        tune_dir=Path("models/ray2").resolve(),
        data_dir=presets.datadir.resolve(),
        use_mean=tune.choice([True, False])
    )

# Define model config
def get_attention_search_space():
    return SearchSpaceAttention(
        input_size=13,
        output_size=20, 
        # size_and_heads=tune.choice(SearchSpaceAttention.get_size_and_heads(4, 120, 4)),
        size_and_heads='120_30',

        dropout=tune.uniform(0.0, 0.3),  
        dropout_attention=tune.uniform(0.0, 0.3),  
        num_layers=4,     
        tune_dir=Path("models/ray2").resolve(),
        data_dir=presets.datadir.resolve(),
        use_mean=False
    )

def get_transformer_search_space():
    # creturn SearchSpaceTransformer(
    #     input_size=13,
    #     output_size=20, 
    #     hidden_size=13, 
    #     num_heads=13,
    #     dropout=tune.uniform(0.0, 0.3),   
    #     num_layers=tune.randint(2, 8),     
    #     tune_dir=Path("models/ray2").resolve(),
    #     data_dir=presets.datadir.resolve(),
    #     use_mean=False,
    #     num_transformer_layers = tune.randint(4, 8)
    # )
    return SearchSpaceTransformer(
        tune_dir=Path("models/ray2").resolve(),
        data_dir=presets.datadir.resolve(),

        # GRU
        input_size=13,
        dropout_gru=tune.uniform(0.0, 0.3),   
        num_layers=tune.randint(2, 6), 
        hidden_size=128,  

        # TransformerEncoderLayer
        num_heads=4,

        # TransformerEncoder
        num_transformer_layers = tune.randint(2, 6),

        # Linear
        output_size=20,     

        # Forward
        use_mean=False     
    )

def get_gru_transformer_search_space():
    return SearchSpaceGRUTransformer(
        tune_dir=Path("models/ray2").resolve(),
        data_dir=presets.datadir.resolve(),

        # GRU
        input_size=13,
        dropout_gru=tune.uniform(0.0, 0.3),   
        num_layers=tune.randint(2, 6), 
        hidden_sizes=13,        

        # TransformerEncoderLayer
        num_heads=13,
        dropout=tune.uniform(0.0, 0.3),   

        # TransformerEncoder
        num_transformer_layers = tune.randint(2, 6),

        # Linear
        output_size=20,     

        # Forward
        use_mean=False     
    )

def get_hyperband_for_bohb_scheduler():
    return HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=200,
        reduction_factor=3,
        stop_last_trials=False,
    )

def get_cli_reporter():
    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    return reporter

def init_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("arabic_digits")

def set_mlflow(model_type):
    mlflow.set_tag("model", model_type)
    mlflow.set_tag("dev", "linksmith")
    mlflow.log_param("datadir", f"{presets.datadir.resolve()}")
    mlflow.log_param("batchsize", f"{presets.batchsize}")
    mlflow.log_params(config.__dict__)


# Define argument parser
parser = argparse.ArgumentParser(description='Model Type for Hyperparameter Tuning')
parser.add_argument('--model_type', type=str, required=True, 
                    help='Type of model to use for hyperparameter tuning. Options: "GRU", "LSTM", "GRUAttention", "Transformer", "GRUTransformer"')
parser.add_argument('--epochs', type=int, required=False, default=50,
                    help='Number of epochs for training.')
args = parser.parse_args()


if __name__ == "__main__":
    print("hack to refresh python cache: 007")
    model_type = args.model_type
    epochs = args.epochs

    init_mlflow()

    with mlflow.start_run():     
        ray.init()

        config, model_class = get_config_and_model_class(model_type)

        # Log hyperparameters to MLflow
        set_mlflow(model_type)

        reporter = get_cli_reporter()
        scheduler = get_hyperband_for_bohb_scheduler()
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
            scheduler=scheduler,
            verbose=1,
        )

        ray.shutdown()
