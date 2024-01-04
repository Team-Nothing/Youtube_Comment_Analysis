import os

import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe

import json
import datetime

data_cover = {}

original_comments = []
original_likes = []
original_views = []

cover_comments = []
cover_likes = []
cover_views = []

time_diff = []

for cover in os.listdir(os.path.join("data", "cover")):
    with open(os.path.join("data", "cover", cover, "overview.json"), encoding="utf-8") as f:
        data = json.load(f)
    if "original_video_id" in data:
        data_cover[data["original_video_id"]] = cover

for original in os.listdir(os.path.join("data", "original")):
    with open(os.path.join("data", "original", original, "overview.json"), encoding="utf-8") as f:
        original_data = json.load(f)
        original_comments.append(float(original_data["comment_count"]))
        original_likes.append(float(original_data["like_count"]))
        original_views.append(float(original_data["view_count"]))

    with open(os.path.join("data", "cover", data_cover[original], "overview.json"), encoding="utf-8") as f:
        cover_data = json.load(f)
        cover_comments.append(float(cover_data["comment_count"]))
        cover_likes.append(float(cover_data["like_count"]))
        cover_views.append(float(cover_data["view_count"]))

    original_date = datetime.datetime.strptime(original_data["published_at"], "%Y-%m-%dT%H:%M:%SZ")
    cover_date = datetime.datetime.strptime(cover_data["published_at"], "%Y-%m-%dT%H:%M:%SZ")
    time_diff.append(float((cover_date - original_date).days))

# calculate mean and std
original_comments = torch.tensor(original_comments)
original_likes = torch.tensor(original_likes)
original_views = torch.tensor(original_views)

cover_comments = torch.tensor(cover_comments)
cover_likes = torch.tensor(cover_likes)
cover_views = torch.tensor(cover_views)

time_diff = torch.tensor(time_diff)

# print without scientific notation
torch.set_printoptions(sci_mode=False)

print("original comments mean: ", original_comments.mean())
print("original comments std: ", original_comments.std())
print("original likes mean: ", original_likes.mean())
print("original likes std: ", original_likes.std())
print("original views mean: ", original_views.mean())
print("original views std: ", original_views.std())

print("cover comments mean: ", cover_comments.mean())
print("cover comments std: ", cover_comments.std())
print("cover likes mean: ", cover_likes.mean())
print("cover likes std: ", cover_likes.std())
print("cover views mean: ", cover_views.mean())
print("cover views std: ", cover_views.std())

print("time diff mean: ", time_diff.mean())
print("time diff std: ", time_diff.std())
