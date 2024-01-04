import pytorch_lightning as pl

from model import TagsModel, RandomCommentsModel
import torch
from torch.nn import functional as F


class PlTagsModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TagsModel()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.__step(batch, batch_idx)
        self.log('train_loss_comment', loss[0], prog_bar=True)
        self.log('train_loss_like', loss[1], prog_bar=True)
        self.log('train_loss_view', loss[2], prog_bar=True)

        return (loss[0] + loss[1] + loss[2]) / 3

    def validation_step(self, batch, batch_idx):
        loss = self.__step(batch, batch_idx)
        self.log('val_loss_comment', loss[0], prog_bar=True)
        self.log('val_loss_like', loss[1], prog_bar=True)
        self.log('val_loss_view', loss[2], prog_bar=True)

    def __step(self, batch, batch_idx):
        original_tags, cover_tags, x, y = batch
        y_hat = self.model(original_tags, cover_tags, x)
        loss_comment = self.loss(y_hat[:, 0], y[:, 0])
        loss_like = self.loss(y_hat[:, 1], y[:, 1])
        loss_view = self.loss(y_hat[:, 2], y[:, 2])
        return loss_comment, loss_like, loss_view

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    @staticmethod
    def loss(y_hat, y):
        return F.l1_loss(y_hat, y)

    @staticmethod
    def optimizer(parameters, lr):
        return torch.optim.Adam(parameters, lr=lr)


class PlRandomCommentsModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = RandomCommentsModel()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.__step(batch, batch_idx)
        self.log('train_loss_comment', loss[0], prog_bar=True)
        self.log('train_loss_like', loss[1], prog_bar=True)
        self.log('train_loss_view', loss[2], prog_bar=True)

        return (loss[0] + loss[1] + loss[2]) / 3

    def validation_step(self, batch, batch_idx):
        loss = self.__step(batch, batch_idx)
        self.log('val_loss_comment', loss[0], prog_bar=True)
        self.log('val_loss_like', loss[1], prog_bar=True)
        self.log('val_loss_view', loss[2], prog_bar=True)

    def __step(self, batch, batch_idx):
        original_tags, cover_tags, original_comments, x, y = batch
        y_hat = self.model(original_tags, cover_tags, original_comments, x)
        loss_comment = self.loss(y_hat[:, 0], y[:, 0])
        loss_like = self.loss(y_hat[:, 1], y[:, 1])
        loss_view = self.loss(y_hat[:, 2], y[:, 2])
        return loss_comment, loss_like, loss_view

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    @staticmethod
    def loss(y_hat, y):
        return F.l1_loss(y_hat, y)

    @staticmethod
    def optimizer(parameters, lr):
        return torch.optim.Adam(parameters, lr=lr)
