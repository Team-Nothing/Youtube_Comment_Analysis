import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from dataset import TagsDataset, RandomCommentsDataset


class PlTagsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, val_split=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.val_split = val_split

        self.train_set, self.val_set = None, None

    def setup(self, stage=None):
        data_set = TagsDataset(self.data_dir)
        val_len = int(len(data_set) * self.val_split)
        train_len = len(data_set) - val_len
        self.train_set, self.val_set = random_split(data_set, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=2, num_workers=6, shuffle=True, persistent_workers=True, collate_fn=TagsDataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=2, num_workers=6, persistent_workers=True, collate_fn=TagsDataset.collate_fn)


class PlRandomCommentsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, val_split=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.val_split = val_split

        self.train_set, self.val_set = None, None

    def setup(self, stage=None):
        data_set = RandomCommentsDataset(self.data_dir)
        val_len = int(len(data_set) * self.val_split)
        train_len = len(data_set) - val_len
        self.train_set, self.val_set = random_split(data_set, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=4, num_workers=6, shuffle=True, persistent_workers=True, collate_fn=RandomCommentsDataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=4, num_workers=6, persistent_workers=True, collate_fn=RandomCommentsDataset.collate_fn)
