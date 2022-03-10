from unittest import main
import torch.nn as nn
from rtdl import FTTransformer
import torch

class Discriminator(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = FTTransformer.make_default(
            n_num_features=args.num_size,
            cat_cardinalities=args.cats,
            d_out=1,
        )




    def forward(self, x_num, x_cat):

        x = self.model.forward(x_num=x_num, x_cat=x_cat)
        return x

