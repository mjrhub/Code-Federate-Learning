#https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test6_mobilenet/model_v2.py

from torch import nn
import torch
from collections import OrderedDict
from operator import itemgetter
from torchstat import stat
from torchsummary import summary
import settings



def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        #groups=1普通卷积
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(in_channel, out_channel,
                                kernel_size, stride, padding, groups=groups, bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU6(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


#倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):#expand_ratio扩展因子
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


def get_subdic_from_dict(base,keys):
    # need make sure keys in dict_key
    od = OrderedDict()
    values = itemgetter(*keys)(base)
    for i, k in enumerate(keys):
        od[k] = values[i]
    return od


def createWholeModel():
    # InvertedResidual: (in_channel, out_channel, stride, expand_ratio)
    # InvertedResidual = 1+2+3+4+3+3+1 = 17
    input_channel = _make_divisible(32 * settings.alpha, settings.round_nearest)
    output_channel1 = _make_divisible(16 * settings.alpha, settings.round_nearest)
    output_channel2 = _make_divisible(24 * settings.alpha, settings.round_nearest)
    output_channel3 = _make_divisible(32 * settings.alpha, settings.round_nearest)
    output_channel4 = _make_divisible(64 * settings.alpha, settings.round_nearest)
    output_channel5 = _make_divisible(96 * settings.alpha, settings.round_nearest)
    output_channel6 = _make_divisible(160 * settings.alpha, settings.round_nearest)
    output_channel7 = _make_divisible(320 * settings.alpha, settings.round_nearest)
    last_channel = _make_divisible(1280 * settings.alpha, settings.round_nearest)
    backbone_layers = OrderedDict([
        ('l1', ConvBNReLU(in_channel=3, out_channel=input_channel, stride=2)),
        ('l2', InvertedResidual(in_channel=input_channel, out_channel=output_channel1, stride=1, expand_ratio=1)),
        ('l3', InvertedResidual(in_channel=output_channel1, out_channel=output_channel2, stride=2,expand_ratio=6)),
        ('l4', InvertedResidual(in_channel=output_channel2, out_channel=output_channel2,stride=1,expand_ratio=6)),
        ('l5', InvertedResidual(in_channel=output_channel2, out_channel=output_channel3, stride=2,expand_ratio=6)),
        ('l6', InvertedResidual(in_channel=output_channel3, out_channel=output_channel3, stride=1, expand_ratio=6)),
        ('l7', InvertedResidual(in_channel=output_channel3, out_channel=output_channel3, stride=1,expand_ratio=6)),
        ('l8', InvertedResidual(in_channel=output_channel3, out_channel=output_channel4, stride=2,expand_ratio=6)),
        ('l9', InvertedResidual(in_channel=output_channel4, out_channel=output_channel4, stride=1,expand_ratio=6)),
        ('l10', InvertedResidual(in_channel=output_channel4, out_channel=output_channel4, stride=1, expand_ratio=6)),
        ('l11', InvertedResidual(in_channel=output_channel4, out_channel=output_channel4, stride=1, expand_ratio=6)),
        ('l12', InvertedResidual(in_channel=output_channel4, out_channel=output_channel5, stride=1,expand_ratio=6)),
        ('l13', InvertedResidual(in_channel=output_channel5, out_channel=output_channel5, stride=1,expand_ratio=6)),
        ('l14', InvertedResidual(in_channel=output_channel5, out_channel=output_channel5, stride=1,expand_ratio=6)),
        ('l15', InvertedResidual(in_channel=output_channel5, out_channel=output_channel6, stride=2,expand_ratio=6)),
        ('l16', InvertedResidual(in_channel=output_channel6, out_channel=output_channel6, stride=1,expand_ratio=6)),
        ('l17', InvertedResidual(in_channel=output_channel6, out_channel=output_channel6, stride=1,expand_ratio=6)),
        ('l18', InvertedResidual(in_channel=output_channel6, out_channel=output_channel7, stride=1,expand_ratio=6)),
        ('l19', ConvBNReLU(in_channel=output_channel7, out_channel=last_channel, kernel_size=1))
    ])
    return backbone_layers


def createClientModel(num_layer):
    x = createWholeModel()
    submodel = nn.Sequential(get_subdic_from_dict(x, list(x.keys())[:num_layer]))
    return submodel


def createServerModel(num_layer):
    x = createWholeModel()
    submodel = nn.Sequential(get_subdic_from_dict(x, list(x.keys())[num_layer:]))
    return submodel


class Mobilenetv2_server(nn.Module):
    def __init__(self, num_train_layer=4):
        super(Mobilenetv2_server, self).__init__()
        self.num = num_train_layer
        self.seq = createServerModel(num_train_layer)
        last_channel = _make_divisible(1280 * settings.alpha, settings.round_nearest)
        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Mobilenetv2_client(nn.Module):
    def __init__(self, num_train_layer=4):
        super(Mobilenetv2_client, self).__init__()
        self.num = num_train_layer
        self.seq = createClientModel(num_train_layer)

    def forward(self, x):
        return self.seq(x)


class Mobilenetv2_whole(nn.Module):
    def __init__(self):
        super(Mobilenetv2_whole, self).__init__()
        self.seq = nn.Sequential(createWholeModel())
        last_channel = _make_divisible(1280 * settings.alpha, settings.round_nearest)
        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = Mobilenetv2_whole().to(device)
# print(net)
# summary(net, (3, 32, 32))
# stat(net, (3, 32, 32))