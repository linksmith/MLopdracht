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
            dropout=config["dropout_1"],
            batch_first=True,
            num_layers=config["num_layers"],
        )

        self.linear = nn.Linear(
            config["hidden_size"], 
            config["output_size"]
        )

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
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
            dropout=config["dropout_1"],
            batch_first=True,
            num_layers=config["num_layers"],
        )

        self.linear = nn.Linear(
            config["hidden_size"], 
            config["output_size"]
        )

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]  # Take the last step
        yhat = self.linear(last_step)
        return yhat


class Attention(nn.Module):
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

        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=config["num_heads"],
            dropout=config["dropout_1"],
            batch_first=True,
        )

        self.linear2 = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]  # Take the last step
        yhat = self.linear2(last_step)
        return yhat


class GRUAttention(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        
        self.rnn = nn.GRU(
            input_size=config["input_size"],
            dropout=config["dropout_1"],
            num_layers=config["num_layers"],
            hidden_size=config["hidden_size"],
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=config["num_heads"],
            dropout=config["dropout_2"],
            batch_first=True,
        )

        self.linear = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
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

        self.transformer = nn.Transformer(
            d_model=config["hidden_size"],
            nhead=config["num_heads"],
            num_encoder_layers=config["num_transformer_layers"],
            num_decoder_layers=config["num_transformer_layers"],
            dim_feedforward=config["hidden_size"] * config["dim_feedforward_multiplier"] ,
            dropout=config["dropout_1"],
            activation='relu',
            batch_first=True
        )

        self.linear2 = nn.Linear(
            config["hidden_size"], 
            config["output_size"]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        batch_size, sequence_length, _ = x.size()
        tgt = torch.zeros((batch_size, sequence_length, self.config["hidden_size"]), device=x.device)
        x = self.transformer(x, tgt=tgt)
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
            dropout=config["dropout_1"],
            num_layers=config["num_layers"],
            hidden_size=config["hidden_size"],
            batch_first=True,
        )

        self.transformer = nn.Transformer(
            d_model=config["hidden_size"],
            nhead=config["num_heads"],
            num_encoder_layers=config["num_transformer_layers"],
            num_decoder_layers=config["num_transformer_layers"],
            dim_feedforward=config["hidden_size"] * config["dim_feedforward_multiplier"] ,
            dropout=config["dropout_2"],
            activation='relu',
            batch_first=True
        )

        self.linear = nn.Linear(
            config["hidden_size"], 
            config["output_size"]
        )

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        batch_size, sequence_length, _ = x.size()
        tgt = torch.zeros((batch_size, sequence_length, self.config["hidden_size"]), device=x.device)
        x = self.transformer(x, tgt=tgt)
        last_step = x[:, -1, :]  # Take the last step
        yhat = self.linear(last_step)
        return yhat
