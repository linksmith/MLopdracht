from typing import Dict

import torch
from torch import nn

Tensor = torch.Tensor


class BaseModel(nn.Module):
    def __init__(self, observations: int, horizon: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()  # we have 3d data, the linear model wants 2D
        self.linear = nn.Linear(observations, horizon)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return x


class BaseRNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, horizon: int
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(hidden_size, horizon)
        self.horizon = horizon

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class GRUmodel(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(
            config["hidden_size"], 
            config["output_size"]
        )


    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)

        if self.config["use_mean"]:
            last_step = x.mean(dim=1)  # Take the mean along the second axis (0-based indexing)
        else:
            last_step = x[:, -1, :]  # Take the last step

        yhat = self.linear(last_step)
        return yhat


class LSTMmodel(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.config = config

        self.rnn = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(
            config["hidden_size"], 
            config["output_size"]
        )


    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)

        if self.config["use_mean"]:
            last_step = x.mean(dim=1)  # Take the mean along the second axis (0-based indexing)
        else:
            last_step = x[:, -1, :]  # Take the last step

        yhat = self.linear(last_step)
        return yhat
    

class GRUAttention(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=4,
            dropout=config["dropout"],
            batch_first=True,
        )
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class AttentionAarabic(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        config["hidden_size"], config["num_heads"] = map(int, config['size_and_heads'].split('_'))
        self.config = config

        self.rnn = nn.GRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=config["num_heads"],
            dropout=config["dropout_attention"],
            batch_first=True,
        )
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        if self.config["use_mean"]:
            last_step = x.mean(dim=1)  # Take the mean along the second axis (0-based indexing)
        else:
            last_step = x[:, -1, :]  # Take the last step

        yhat = self.linear(last_step)
        return yhat


class TransformerAarabic(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()        
        self.config = config

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["input_size"], 
            nhead=config["num_heads"],
            dropout=config["dropout"],
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=config["num_transformer_layers"]
        )

        self.linear = nn.Linear(
            config["hidden_size"], 
            config["output_size"]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.transformer_encoder(x)

        if self.config["use_mean"]:
            last_step = x.mean(dim=1)  # Take the mean along the second axis (0-based indexing)
        else:
            last_step = x[:, -1, :]  # Take the last step

        yhat = self.linear(last_step)
        return yhat
    

class GRUTransformerAarabic(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()        
        self.config = config

        self.rnn = nn.GRU(
            input_size=config["input_size"],
            dropout=config["dropout_gru"],
            num_layers=config["num_layers"],
            batch_first=True,
            hidden_size=config["hidden_sizes"],
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden_sizes"], 
            nhead=config["num_heads"],
            dropout=config["dropout"],
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=config["num_transformer_layers"]
        )

        self.linear = nn.Linear(
            config["hidden_sizes"], 
            config["output_size"]
        )

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x = self.transformer_encoder(x)

        if self.config["use_mean"]:
            last_step = x.mean(dim=1)  # Take the mean along the second axis (0-based indexing)
        else:
            last_step = x[:, -1, :]  # Take the last step

        yhat = self.linear(last_step)
        return yhat
    

class NLPmodel(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(config["vocab"], config["hidden_size"])
        self.rnn = nn.GRU(
            input_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class AttentionNLP(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(config["vocab"], config["hidden_size"])
        self.rnn = nn.GRU(
            input_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=4,
            dropout=config["dropout"],
            batch_first=True,
        )
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat
