import json
import os.path
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
import torch.nn.functional as F
import datetime


class TagsDataset(Dataset):
    def __init__(self, data_root, comments=100):
        self.comments = comments
        self.glove = GloVe(name='twitter.27B', dim=50)
        self.tokenizer = get_tokenizer("basic_english")
        self.data_original = os.listdir(os.path.join(data_root, "original"))
        self.data_cover = {}
        for cover in os.listdir(os.path.join(data_root, "cover")):
            with open(os.path.join(data_root, "cover", cover, "overview.json"), encoding="utf-8") as f:
                data = json.load(f)
            if "original_video_id" in data:
                self.data_cover[data["original_video_id"]] = cover

    def __len__(self):
        return len(self.data_original)

    def __getitem__(self, index):
        original = self.data_original[index]
        cover = self.data_cover[original]

        with open(os.path.join("data", "original", original, "overview.json"), encoding="utf-8") as f:
            original_data = json.load(f)
        with open(os.path.join("data", "cover", cover, "overview.json"), encoding="utf-8") as f:
            cover_data = json.load(f)

        if "tags" in original_data and original_data["tags"] is not None:
            original_tags = [self.glove.get_vecs_by_tokens(self.tokenizer(i), lower_case_backup=True) for i in original_data["tags"]]
            original_tags = self.__padding_tags(original_tags)
        else:
            original_tags = torch.zeros((1, 5, 50))

        if "tags" in cover_data and cover_data["tags"] is not None:
            cover_tags = [self.glove.get_vecs_by_tokens(self.tokenizer(i), lower_case_backup=True) for i in cover_data["tags"]]
            cover_tags = self.__padding_tags(cover_tags)
        else:
            cover_tags = torch.zeros((1, 5, 50))

        original_date = datetime.datetime.strptime(original_data["published_at"], "%Y-%m-%dT%H:%M:%SZ")
        cover_date = datetime.datetime.strptime(cover_data["published_at"], "%Y-%m-%dT%H:%M:%SZ")

        # normalize with mean and ste

        x = torch.tensor([
            ((cover_date - original_date).days - 642.6000) / 1052.6399,
            (float(original_data["comment_count"]) - 67520.2266) / 104567.2812,
             (float(original_data["like_count"]) - 2027378.75) / 3711913,
              (float(original_data["view_count"]) - 277567488) / 100000,
        ], dtype=torch.float)

        y = torch.tensor([
            (float(cover_data["comment_count"]) - 587.7143) / 966.7789,
            (float(cover_data["like_count"]) - 21659.9570) / 37724.0508,
            (float(cover_data["view_count"]) - 831191.2500) / 1228311.2500,
        ], dtype=torch.float)

        return original_tags, cover_tags, x, y

    @staticmethod
    def collate_fn(batch):
        original_tags, cover_tags, x, y = zip(*batch)
        original_tags = pad_sequence(original_tags, batch_first=True)
        cover_tags = pad_sequence(cover_tags, batch_first=True)
        x = torch.stack(x)
        y = torch.stack(y)
        return original_tags, cover_tags, x, y

    @staticmethod
    def __padding_tags(tags):
        max_length = 5
        processed_tags = []
        for tag in tags:
            current_length = tag.size(0)
            if current_length < max_length:
                padded_tag = F.pad(tag, (0, 0, 0, max_length - current_length))
            else:
                padded_tag = tag[:max_length, :]

            processed_tags.append(padded_tag)
        return torch.stack(processed_tags)


class RandomCommentsDataset(Dataset):
    def __init__(self, data_root, comments=100):
        self.comments = comments
        self.glove = GloVe(name='twitter.27B', dim=50)
        self.tokenizer = get_tokenizer("basic_english")
        self.data_original = os.listdir(os.path.join(data_root, "original"))
        self.data_cover = {}
        for cover in os.listdir(os.path.join(data_root, "cover")):
            with open(os.path.join(data_root, "cover", cover, "overview.json"), encoding="utf-8") as f:
                data = json.load(f)
            if "original_video_id" in data:
                self.data_cover[data["original_video_id"]] = cover

    def __len__(self):
        return len(self.data_original)

    def __getitem__(self, index):
        original = self.data_original[index]
        cover = self.data_cover[original]

        with open(os.path.join("data", "original", original, "overview.json"), encoding="utf-8") as f:
            original_data = json.load(f)
        with open(os.path.join("data", "cover", cover, "overview.json"), encoding="utf-8") as f:
            cover_data = json.load(f)

        if "tags" in original_data and original_data["tags"] is not None:
            original_tags = [self.glove.get_vecs_by_tokens(self.tokenizer(i), lower_case_backup=True) for i in original_data["tags"]]
            original_tags = self.__padding_tags(original_tags)
        else:
            original_tags = torch.zeros((1, 5, 50))

        if "tags" in cover_data and cover_data["tags"] is not None:
            cover_tags = [self.glove.get_vecs_by_tokens(self.tokenizer(i), lower_case_backup=True) for i in cover_data["tags"]]
            cover_tags = self.__padding_tags(cover_tags)
        else:
            cover_tags = torch.zeros((1, 5, 50))

        original_comments = []
        comment_paths = os.listdir(os.path.join("data", "original", original))[:-1]

        random.shuffle(comment_paths)
        for comment in comment_paths[:200]:
            with open(os.path.join("data", "original", original, comment), encoding="utf-8") as f:
                data = json.load(f)
            try:
                original_comments.append(
                    self.glove.get_vecs_by_tokens(self.tokenizer(data["text"]), lower_case_backup=True))
            except:
                continue
        original_comments = self.__padding_tags(original_comments, 100)

        original_date = datetime.datetime.strptime(original_data["published_at"], "%Y-%m-%dT%H:%M:%SZ")
        cover_date = datetime.datetime.strptime(cover_data["published_at"], "%Y-%m-%dT%H:%M:%SZ")

        x = torch.tensor([
            (cover_date - original_date).days / 10000,
            float(original_data["comment_count"]) / 10000,
            float(original_data["like_count"]) / 100000,
            float(original_data["view_count"]) / 1000000,
        ], dtype=torch.float)

        y = torch.tensor([
            float(cover_data["comment_count"]) / 10000,
            float(cover_data["like_count"]) / 100000,
            float(cover_data["view_count"]) / 1000000,
        ], dtype=torch.float)

        return original_tags, cover_tags, original_comments, x, y

    @staticmethod
    def collate_fn(batch):
        original_tags, cover_tags, original_comments, x, y = zip(*batch)
        original_tags = pad_sequence(original_tags, batch_first=True)
        cover_tags = pad_sequence(cover_tags, batch_first=True)
        original_comments = pad_sequence(original_comments, batch_first=True)
        x = torch.stack(x)
        y = torch.stack(y)
        return original_tags, cover_tags, original_comments, x, y

    @staticmethod
    def __padding_tags(tags, max_length=5):
        processed_tags = []
        for tag in tags:
            current_length = tag.size(0)
            if current_length < max_length:
                padded_tag = F.pad(tag, (0, 0, 0, max_length - current_length))
            else:
                padded_tag = tag[:max_length, :]

            processed_tags.append(padded_tag)
        return torch.stack(processed_tags)

