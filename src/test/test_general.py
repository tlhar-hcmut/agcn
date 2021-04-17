import time
import unittest

import torch


class TestGeneral(unittest.TestCase):
    def test(self):
        start = time.time()
        for i in range(1000):
            pos_dnn(1000, 1000)
        end1 = time.time()
        for i in range(1000):
            pos_me(1000, 1000)
        end2 = time.time()

        print(start)
        print(end1)
        print(end2)
        print(end1 - start)
        print(end2 - end1)


def pos_dnn(num_hiddens, max_len=1000):
    # Create a long enough P
    P = torch.zeros((1, max_len, num_hiddens))
    X = torch.arange(0, max_len).reshape(-1, 1) / torch.pow(
        10000, torch.arange(0, num_hiddens, 2) / num_hiddens
    )
    P[:, :, 0::2] = torch.sin(X)
    P[:, :, 1::2] = torch.cos(X)
    return P


def pos_me(T, F):
    mat_idx = (
        torch.arange(0, T, 1, dtype=torch.float)
        .unsqueeze(0)
        .repeat(F, 1)
        .transpose(0, 1)
    )

    mat_idx_F_x = torch.arange(0, F, 2).unsqueeze(0).repeat(T, 1)
    mat_idx_F_y = (
        torch.arange(0, T, 1).unsqueeze(0).repeat((F + 1) // 2, 1).transpose(0, 1)
    )

    mat_idx[mat_idx_F_y, mat_idx_F_x] = torch.sin(
        mat_idx_F_y / (10000 ** ((mat_idx_F_x) / F))
    )
    mat_idx[mat_idx_F_y[:, :-1], mat_idx_F_x[:, :-1] + 1] = torch.cos(
        mat_idx_F_y[:, :-1] / (10000 ** ((mat_idx_F_x[:, :-1]) / F))
    )
    if F % 2 == 0:
        A = mat_idx_F_y[:, -1:]
        B = mat_idx_F_x[:, -1:] + 1
        C = mat_idx_F_x[:, -1:]
        mat_idx[A, B] = torch.cos(A / (10000 ** ((C) / F)))
    return mat_idx
