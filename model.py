import torch
from torch import nn


class TagsModel(nn.Module):
    def __init__(self, tag_words=5, tag_dim=50):
        super(TagsModel, self).__init__()
        self.tag_words = tag_words
        self.tag_dim = tag_dim

        self.lstm_a = nn.LSTM(tag_dim * tag_words, 20, batch_first=True)
        self.lstm_b = nn.LSTM(tag_dim * tag_words, 20, batch_first=True)

        self.linear = nn.Linear(44, 3)

    def forward(self, tags_a, tags_b, x):
        tags_a = tags_a.view(tags_a.shape[0], tags_a.shape[1], self.tag_dim * self.tag_words)
        tags_b = tags_b.view(tags_b.shape[0], tags_b.shape[1], self.tag_dim * self.tag_words)

        tags_a, _ = self.lstm_a(tags_a)
        tags_b, _ = self.lstm_b(tags_b)

        tags_a = tags_a[:, -1, :]
        tags_b = tags_b[:, -1, :]

        x = torch.cat((tags_a, tags_b, x), dim=1)
        x = self.linear(x)

        return x


class RandomCommentsModel(nn.Module):
    def __init__(self, tag_words=5, tag_dim=50):
        super(RandomCommentsModel, self).__init__()
        self.tag_words = tag_words
        self.tag_dim = tag_dim

        self.lstm_a = nn.LSTM(tag_dim * tag_words, 20, batch_first=True)
        self.lstm_b = nn.LSTM(tag_dim * tag_words, 20, batch_first=True)

        self.lstm_c = nn.LSTM(tag_dim * 100, 300, batch_first=True)

        self.linear = nn.Linear(344, 3)

    def forward(self, tags_a, tags_b, comments, x):
        tags_a = tags_a.view(tags_a.shape[0], tags_a.shape[1], self.tag_dim * self.tag_words)
        tags_b = tags_b.view(tags_b.shape[0], tags_b.shape[1], self.tag_dim * self.tag_words)
        comments = comments.view(comments.shape[0], comments.shape[1], self.tag_dim * 100)

        tags_a, _ = self.lstm_a(tags_a)
        tags_b, _ = self.lstm_b(tags_b)
        comments, _ = self.lstm_c(comments)

        tags_a = tags_a[:, -1, :]
        tags_b = tags_b[:, -1, :]
        comments = comments[:, -1, :]

        x = torch.cat((tags_a, tags_b, comments, x), dim=1)
        x = self.linear(x)

        return x
