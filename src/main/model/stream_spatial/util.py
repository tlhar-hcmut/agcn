import math

import torch.nn as nn


def init_conv_branch(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def init_conv(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def init_bn(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
