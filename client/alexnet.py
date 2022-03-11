# Alexnet
import torch
import torch.nn as nn
import copy
from collections import OrderedDict
from operator import itemgetter


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=False):
        super(ConvBlock, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, verbose=False):
        out = self.block(x)
        if verbose:
            print(x.shape, "->", out.shape)
        return out


class Classifier(nn.Module):
    """Fully connected classifier module."""

    def __init__(self, in_features, middle_features=4096, out_features=10, n_fc_layers=3):
        super(Classifier, self).__init__()
        layers = list()
        '''说明：
            bool（）函数，如果遇到空的列表，即0，返回是False，用来判断一个对象是否为空，
            空就是False，有东西就是1，返回True

            not 函数，正相反，如果遇到空的列表，返回值是True，说明这是一个空值'''

        is_last_layer = not bool(n_fc_layers)
        layers.append(nn.Linear(in_features=in_features,
                                out_features=out_features if is_last_layer else middle_features))

        n_fc_layers -= 1
        while n_fc_layers > 0:
            is_last_layer = n_fc_layers <= 1
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.5))
            layers.append(nn.Linear(in_features=middle_features,
                                    out_features=out_features if is_last_layer else middle_features))
            n_fc_layers -= 1
        self.fc = nn.Sequential(*layers)

    def forward(self, x, verbose=False):
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        if verbose:
            print(x.shape, "->", out.shape)
        return out


def createWholeModel():
    backbone_layers = OrderedDict([
        ('l1', ConvBlock(in_channels=3, out_channels=64, kernel_size=3, stride=2)),
        ('l2', nn.MaxPool2d(kernel_size=2)),
        ('l3', ConvBlock(in_channels=64, out_channels=192)),
        ('l4', nn.MaxPool2d(kernel_size=2)),
        ('l5', ConvBlock(in_channels=192, out_channels=384)),
        ('l6', ConvBlock(in_channels=384, out_channels=256)),
        ('l7', ConvBlock(in_channels=256, out_channels=256)),
        ('l8', nn.MaxPool2d(kernel_size=2))
    ])
    return backbone_layers


class Alexnet_whole(nn.Module):
    def __init__(self):
        super(Alexnet_whole, self).__init__()
        self.seq = nn.Sequential(createWholeModel())
        # self.classifier = Classifier(in_features=512)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        # batch_size = x.shape[0]
        # x = x.view(batch_size, -1)
        # x = x.view(x.size(0), -1)
        # x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), 256 * 2 * 2)
        out = self.classifier(x)
        return out


# test
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# wholeModel = Alexnet_whole().to(device)
# print(wholeModel)
# summary(wholeModel, (3, 32, 32))
# stat(wholeModel, (3, 32, 32))