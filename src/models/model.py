from typing import Dict
import torch
from torch import nn

Tensor = torch.Tensor


class GRU(nn.Module):
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


class LSTM(nn.Module):
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
        config["hidden_size"], config["num_heads"] = map(int, config['size_and_heads'].split('_'))
        
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


class Attention(nn.Module):
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


class Transformer(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()        
        self.config = config

        self.linear1 = nn.Linear(
            config["input_size"], 
            config["hidden_size"]
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden_size"], 
            nhead=config["num_heads"],
            dropout=config["dropout"],
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=config["num_transformer_layers"]
        )

        self.linear2 = nn.Linear(
            config["hidden_size"], 
            config["output_size"]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.transformer_encoder(x)

        if self.config["use_mean"]:
            last_step = x.mean(dim=1)  # Take the mean along the second axis (0-based indexing)
        else:
            last_step = x[:, -1, :]  # Take the last step

        yhat = self.linear2(last_step)
        return yhat
    

class GRUTransformer(nn.Module):
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


class TestGRUTransformer(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()        

        self.rnn = nn.GRU(
            input_size=13,
            dropout=0.1,
            num_layers=2,
            batch_first=True,
            hidden_size=13,
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=13, 
            nhead=13,
            dropout=0.2,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=4
        )

        self.linear = nn.Linear(
            13, 
            20
        )

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)

        x, _ = self.rnn(x)
        x = self.transformer_encoder(x)
        last_step = x[:, -1, :]  # Take the last step
        yhat = self.linear(last_step)
        return yhat
