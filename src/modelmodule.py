import pytorch_lightning as pl
import torch
from torch import nn
from .model import SamplePytorchModel


class MyParameters:
    max_epochs: int = 30
    early_stopping: bool = True
    early_stopping_patience: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001


class SampleLitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.params = MyParameters()
        self.model = SamplePytorchModel()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
        return optimizer

    def training_step(self, batch, batch_index):
        """
        各バッチに対して適用する学習処理
        """
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": (pred.argmax(1) == y).sum().type(torch.float),
            },
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_index):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": (pred.argmax(1) == y).sum().type(torch.float),
            },
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_index):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": (pred.argmax(1) == y).sum().type(torch.float),
            },
            on_epoch=True,
        )
        return loss

    def on_train_epoch_start(self):
        self.print(f"Epoch {self.trainer.current_epoch+1}/{self.trainer.max_epochs}")
        return super().on_train_epoch_start()

    def on_train_start(self):
        self.print("Start training.")
        return super().on_train_start()

    def on_train_end(self):
        self.print("End training.")
        return super().on_train_end()
