
import torch.nn as nn
import functools
from layers import *


class Decoder(nn.Module):
    def __init__(self, batch_size, num_classes, h_dim=259, z_dim=128, norm_layer=None):
        super(Decoder, self).__init__()
        if norm_layer is None:
            bn = functools.partial(ccbn,
                                   which_linear=(functools.partial(nn.Linear, bias=False)),
                                   cross_replica=False,
                                   mybn=False,
                                   input_size=num_classes)
            self.norm_layer = bn
        self.h_dim, self.z_dim = h_dim, z_dim
        self.ch = 256
        self.relu = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(self.ch, self.ch * 2)
        self.bn6 = self.norm_layer(self.ch * 2, momentum=0.01)

    def forward(self, z, y):
        x = self.linear1(z)
        x = self.bn6(x, y)
        x = self.relu(x)

        return x