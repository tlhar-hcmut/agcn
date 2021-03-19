import math

import numpy as np
from tqdm import tqdm
from xcommon import xconsole


def normalize(data: np.ndarray, zaxis=[0, 1], xaxis=[8, 4], silent=False) -> np.ndarray:
    """
    Normalize skeleton
    - input: N, C, T, V, M
    - output: N, C, T, V, M
    """
    data = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C
    pad_null_frame(data, silent)
    sub_center_joint(data, silent)
    align_vertical(data, zaxis, silent)
    align_horizontal(data, xaxis, silent)
    return np.transpose(data, [0, 4, 2, 3, 1])


def pad_null_frame(data: np.ndarray, silient=False) -> None:
    """
    Pad the null frames with the previous frames
    - input: N, M, T, V, C
    """
    for idx_s, sample in enumerate(tqdm(data, disable=silient)):
        if sample.sum() == 0:
            xconsole.info(idx_s + " has no data!")
        for idx_b, body in enumerate(sample):
            if body.sum() == 0:
                continue
            index = body.sum(-1).sum(-1) != 0
            tmp = body[index].copy()
            body *= 0
            body[: len(tmp)] = tmp
            for idx_f, frame in enumerate(body):
                if frame.sum() == 0:
                    rest = len(body) - idx_f
                    num = int(np.ceil(rest / idx_f))
                    pad = np.concatenate([body[0:idx_f] for _ in range(num)], 0)[:rest]
                    data[idx_s, idx_b, idx_f:] = pad
                    break


def sub_center_joint(data: np.ndarray, silient=False) -> None:
    """
    Sub the center joint #1 (spine joint in ntu dataset
    - input: N, M, T, V, C
    """
    N, M, T, V, C = data.shape

    for i_s, sample in enumerate(tqdm(data, disable=silient)):
        if sample.sum() == 0:
            continue
        main_body_center = sample[0][:, 1:2, :].copy()
        for i_b, body in enumerate(sample):
            if body.sum() == 0:
                continue
            mask = (body.sum(-1) != 0).reshape(T, V, 1)
            data[i_s, i_b] = (data[i_s, i_b] - main_body_center) * mask


def align_vertical(data: np.ndarray, zaxis=[0, 1],  silient=False) -> None:
    """
    parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis
    - input: N, M, T, V, C
    """
    for i_s, skeleton in enumerate(tqdm(data, disable=silient)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]  # hip(jpt 0)
        joint_top = skeleton[0, 0, zaxis[1]]  # spine(jpt 1)
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = get_angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotate_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    data[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)


def align_horizontal(data: np.ndarray, xaxis=[8, 4],  silient=False) -> None:
    """
    parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis
    - input: N, M, T, V, C
    """
    for i_s, skeleton in enumerate(tqdm(data, disable=silient)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = get_angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotate_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    data[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)


def rotate_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def cal_unit_vec(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = cal_unit_vec(v1)
    v2_u = cal_unit_vec(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
