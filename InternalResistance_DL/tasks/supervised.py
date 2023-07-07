import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
import csv


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        feat_max_val_label: float = 1.0,
        feat_max_val2: float = 1.0,
        feat_max_val_label2: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self.regressor2 = nn.Linear(
                6,
                self.hparams.pre_len,
            )
        self._loss = loss
        self.feat_max_val = feat_max_val
        self.feat_max_val_label = feat_max_val_label
        self.feat_max_val2 = feat_max_val2
        self.feat_max_val_label2 = feat_max_val_label2

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        #hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)

        if self.regressor is not None:
            predictions = self.regressor(hidden)
            predictions=(predictions.squeeze(-1))
        else:
            predictions = hidden
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y,ID,T = batch
        self.ID=ID
        #predictions = self(x)
        predictions=ID
        #predictions=torch.cat((predictions,ID),dim=-1)
        #predictions = torch.cat((predictions, T),dim=-1)
        predictions=self.regressor2(predictions)
        y=y.unsqueeze(-1)
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val_label2
        y = y * self.feat_max_val_label2
        loss = self.loss(predictions, y)
        self.log("val_loss", loss)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        mask = torch.gt(y, 0)
        pred = torch.masked_select(predictions, mask)
        true = torch.masked_select(y, mask)
        mape = torch.mean(torch.abs(torch.div((true - pred), true)))
        mape_max=torch.max(torch.abs(torch.div((true - pred), true)))
        mape_min = torch.min(torch.abs(torch.div((true - pred), true)))
        explained_variance = utils.metrics.explained_variance(predictions, y)
        ID_char=''
        for ch in (self.ID[0]):
            ID_char+=chr(int(ch))
        print()
        print(ID_char,'RMSE:',rmse,'MAE:',mae,'MAPE:',mape,'max_MAPE:',mape_max,'min_MAPE:',mape_min)
        print()
        metrics = {
            ID_char: 0,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE":mape,
            "max_MAPE":mape_max,
            "min_MAPE":mape_min,
        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val_label2
        y = y * self.feat_max_val_label2
        loss = self.loss(predictions, y)
        pos_bias = 0
        mask = torch.gt(y, 20000 + 500 * pos_bias) ^ torch.gt(y, 20500 + 500 * pos_bias)
        #print(torch.gt(y, 20000 + 500 * pos_bias),'11111')
        #print(torch.gt(y, 20500 + 500 * pos_bias),'22222')
        #print(mask)

        pred = torch.masked_select(predictions, mask)
        true = torch.masked_select(y, mask)
        print(pred,'pred')
        print(true,'true')

        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pred, true))
        mae = torchmetrics.functional.mean_absolute_error(pred, true)
        mape = torch.mean(torch.abs(torch.div((true - pred), true)))
        mape_max = torch.max(torch.abs(torch.div((true - pred), true)))
        mape_min = torch.min(torch.abs(torch.div((true - pred), true)))
        explained_variance = utils.metrics.explained_variance(predictions, y)
        ID_char = ''
        for ch in (self.ID[0]):
            ID_char += chr(int(ch))
        print()
        print(ID_char, 'RMSE:', rmse, 'MAE:', mae, 'MAPE:', mape,'max_MAPE:',mape_max,'min_MAPE:',mape_min)
        print()
        metrics = {
            ID_char: 20000+pos_bias*500,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "max_MAPE": mape_max,
            "min_MAPE": mape_min,
        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=5e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=3e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser
