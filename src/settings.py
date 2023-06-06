from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union, cast
from pydantic import BaseModel, HttpUrl, root_validator
from ray import tune
from src.models.metrics import Metric

cwd = Path(__file__)
root = (cwd / "../..").resolve()

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float
SAMPLE_CATEGORICAL = tune.search.sample.Categorical


class Settings(BaseModel):
    datadir: Path
    testurl: HttpUrl
    trainurl: HttpUrl
    testfile: Path
    trainfile: Path
    modeldir: Path
    logdir: Path
    modelname: str
    batchsize: int


# note pydantic handles perfectly a string as url
# but mypy doesnt know that, so to keep mypy satisfied
# i am adding the "cast" for the urls.
presets = Settings(
    datadir=root / "data/raw",
    testurl=cast(
        HttpUrl,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Test_Arabic_Digit.txt",  # noqa N501
    ),
    trainurl=cast(
        HttpUrl,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt",  # noqa N501
    ),
    testfile=Path("ArabicTest.txt"),
    trainfile=Path("ArabicTrain.txt"),
    modeldir=root / "models",
    logdir=root / "logs",
    modelname="model.pt",
    batchsize=64,
)


class TrainerSettings(BaseModel):
    epochs: int
    metrics: List[Metric]
    logdir: Path
    train_steps: int
    valid_steps: int
    tunewriter: List[str] = ["tensorboard"]
    optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 1e-5}
    scheduler_kwargs: Optional[Dict[str, Any]] = {"factor": 0.1, "patience": 10}
    earlystop_kwargs: Optional[Dict[str, Any]] = {
        "save": False,
        "verbose": True,
        "patience": 10,
    }

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())



class MlFlowTrainerSettings(TrainerSettings):
    modelpath: Path


class BaseSearchSpace(BaseModel):
    input_size: int
    output_size: int
    tune_dir: Optional[Path]
    data_dir: Path

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def check_path(cls, values: Dict) -> Dict:  # noqa: N805
        datadir = values.get("data_dir")
        if not datadir.exists():
            raise FileNotFoundError(
                f"Make sure the datadir exists.\n Found {datadir} to be non-existing."
            )
        return values


# this is what ray will use to create configs
class SearchSpace(BaseSearchSpace):
    hidden_size: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    dropout_1: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.3)
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 5)
    

# this is what ray will use to create configs
class SearchSpaceAttention(SearchSpace):
    num_heads: Union[str, SAMPLE_CATEGORICAL] = tune.choice([2, 4, 8, 16])
    hidden_size: Union[int, SAMPLE_INT] = tune.qrandint(32, 256, 16)


# this is what ray will use to create configs
class SearchSpaceGRUAttention(SearchSpaceAttention):
    dropout_2: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.3)


# this is what ray will use to create configs
class SearchSpaceTransformer(SearchSpaceAttention):
    dim_feedforward_multiplier: Union[int, SAMPLE_INT] = tune.randint(1, 3)
    num_transformer_layers: Union[int, SAMPLE_INT] = tune.randint(4, 8)


# this is what ray will use to create configs
class SearchSpaceGRUTransformer(SearchSpaceTransformer):
    dropout_2: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.3)


class BaseSettings(BaseModel):
    data_dir: Path


cwd = Path(__file__).parent
cwd = (cwd / "../").resolve()
# print('ab')
from datetime import datetime
print(datetime.now())

class GeneralSettings(BaseSettings):
    data_dir = cwd / "data/raw"