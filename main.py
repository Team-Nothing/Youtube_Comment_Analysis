import torch

from pl_data_module import PlTagsDataModule, PlRandomCommentsDataModule
from pl_module import PlTagsModule, PlRandomCommentsModule

import pytorch_lightning as pl

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    data_module = PlRandomCommentsDataModule("data")
    model = PlRandomCommentsModule()
    logger = pl.loggers.TensorBoardLogger("logs", name="comments")

    trainer = pl.Trainer(max_epochs=1000, logger=logger, log_every_n_steps=1)
    trainer.fit(model, data_module)
