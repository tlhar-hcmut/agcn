from torch.nn import init 
import math

def init_conv_branch(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    init.constant_(conv.bias, 0)


def init_conv(conv):
    init.kaiming_normal_(conv.weight, mode="fan_out")
    init.constant_(conv.bias, 0)


def init_bn(bn, scale):
    init.constant_(bn.weight, scale)
    init.constant_(bn.bias, 0)
