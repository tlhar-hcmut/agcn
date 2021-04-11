import unittest

import numpy as np
import src.main.model as M
from src.main.graph import NtuGraph
from torchsummary import summary
import torch
from torch import nn
import timeit


def pos_dnn(num_hiddens, max_len=1000):
    # Create a long enough P
    P = np.zeros((1, max_len, num_hiddens))
    X = np.arange(0, max_len).reshape(-1, 1) / np.power(10000, np.arange(0, num_hiddens, 2) / num_hiddens)
    P[:, :, 0::2] = np.sin(X)
    P[:, :, 1::2] = np.cos(X)
    return P


def pos_me(T, F):
    mat_idx = torch.arange(0, T, 1, dtype=torch.float).unsqueeze(0).repeat(F, 1).transpose(0, 1)

    mat_idx_F_x = torch.arange(0, F, 2).unsqueeze(0).repeat(T, 1)
    mat_idx_F_y = torch.arange(0, T, 1).unsqueeze(0).repeat((F+1)//2, 1).transpose(0, 1)

    mat_idx[mat_idx_F_y, mat_idx_F_x] = torch.sin(mat_idx_F_y/(10000**((mat_idx_F_x)/F)))
    mat_idx[mat_idx_F_y[:, :-1], mat_idx_F_x[:, :-1] + 1] = torch.cos(mat_idx_F_y[:, :-1]/(10000**((mat_idx_F_x[:, :-1])/F)))
    if (F % 2 == 0):
        A=mat_idx_F_y[:, -1:]
        B=mat_idx_F_x[:, -1:]+1
        C=mat_idx_F_x[:, -1:]
        mat_idx[A, B] = torch.cos(A/(10000**((C)/F)))
    return mat_idx


if (__name__ == "__main__"):
    start = timeit.timeit()
    for i in range(1):
        pos_dnn(1000, 1000)
    end1 = timeit.timeit()
    for i in range(1):
        pos_me(1000, 1000)
    end2 = timeit.timeit()

    print(end1-start)
    print(end2-end1)
