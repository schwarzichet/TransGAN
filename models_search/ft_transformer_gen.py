from audioop import bias
import enum
from random import random
import sys
from turtle import forward
import xdrlib
import torch.nn as nn
from rtdl import MultiheadAttention
import torch.nn.functional as F
import torch


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Generator(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.bottom_width = args.bottom_width  #
        self.embed_dim = args.gf_dim

        self.cat_size = args.cat_size
        self.num_size = args.num_size
        # self.cat_num = sum(args.cats)
        self.cats = args.cats

        self.l1 = nn.Linear(args.latent_dim, self.bottom_width * 1 * self.embed_dim)  #
        self.att = MultiheadAttention(
            d_token=self.embed_dim,
            n_heads=8,
            dropout=0.2,
            bias=True,
            initialization="kaiming",
        )
        self.norm = nn.BatchNorm1d(self.embed_dim)
        self.mlp = Mlp(
            in_features=self.embed_dim, hidden_features=4 * self.embed_dim, drop=0.2
        )
        # self.mlp_cat = Mlp(8192, out_features=self.cat_size*self.cat_num)

        self.mlp_cats = []
        for i in self.cats:
            self.mlp_cats.append(Mlp(8192, out_features=i).to("cuda"))

        self.mlp_num = Mlp(8192, out_features=self.num_size)

        self.upscale = torch.nn.PixelShuffle(8)

    def forward(self, random_sample, epoch):
        x = self.l1(random_sample).view(
            -1,
            self.bottom_width * 2,
            self.embed_dim,
        )
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + (self.att(x, x, None, None)[0])
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + (self.mlp(x))

        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + (self.att(x, x, None, None)[0])
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + (self.mlp(x))

        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + (self.att(x, x, None, None)[0])
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + (self.mlp(x))

        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + (self.att(x, x, None, None)[0])
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + (self.mlp(x))
        # print(x.shape)
        x = self.upscale(x)

        # x_cat = self.mlp_cat(x).view(-1, self.cat_size, self.cat_num)
        x_cats = []
        for mlp_model, cat_size in zip(self.mlp_cats, self.cats):
            # print(mlp_model.iscuda)
            x_cats.append(mlp_model(x).view(-1, cat_size))
        x_num = self.mlp_num(x).view(-1, self.num_size)

        x_cats = [F.gumbel_softmax(x, 1, hard=True) for x in x_cats]
        x_cats_padding = []



        for i, t in enumerate(x_cats):
            x_cats_padding.append(
                nn.ZeroPad2d((sum(self.cats[:i]), sum(self.cats[i+1:]), 0, 0))(t)
            )

        # for i in x_cats_padding:
        #     print(i)
        #     print(i.shape)
        # print(x_cats_padding[0].shape)
        # print(torch.stack(x_cats_padding, dim=1).shape)

        # sys.exit()

        return x_num, torch.stack(x_cats_padding, dim=1)
