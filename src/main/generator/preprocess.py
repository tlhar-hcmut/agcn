import numpy as np
from src.main.util import rotation
from tqdm import tqdm
from xcommon import xconsole


def normalize(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    pad_null_frame(s)
    sub_center_joint(s)
  
    

    print("")
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = rotation.get_angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation.rotate_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


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


def erect_skeleton(data: np.ndarray, silient=False) -> None:
    """
    parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) 
    of the first person to the x axis
    - input: N, M, T, V, C
    """
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]  # hip(jpt 0)
        joint_top = skeleton[0, 0, zaxis[1]]  # spine(jpt 1)
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = rotation.get_angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation.rotate_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)
