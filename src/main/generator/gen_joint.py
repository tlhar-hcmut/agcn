import os
import pickle
from typing import *

import numpy as np
from src.main.generator import preprocess
from src.main.util import analyst
from src.main.util.config import config_glob
from tqdm import tqdm


def read_skeleton(file: str) -> Dict:
    """
    Put all infos of .skeleton files  into dictionary "skeleton_sequence"
    """
    with open(file, "r") as f:
        skeleton_sequence = {}
        skeleton_sequence["numFrame"] = int(f.readline())
        skeleton_sequence["frameInfo"] = []
        # num_body = 0
        for _ in range(skeleton_sequence["numFrame"]):
            frame_info = {}
            frame_info["numBody"] = int(f.readline())
            frame_info["bodyInfo"] = []

            for _ in range(frame_info["numBody"]):
                body_info = {}
                body_info_key = [
                    "bodyID",
                    "clipedEdges",
                    "handLeftConfidence",
                    "handLeftState",
                    "handRightConfidence",
                    "handRightState",
                    "isResticted",
                    "leanX",
                    "leanY",
                    "trackingState",
                ]
                body_info = {
                    k: float(v) for k, v in zip(body_info_key, f.readline().split())
                }
                body_info["numJoint"] = int(f.readline())
                body_info["jointInfo"] = []
                for _ in range(body_info["numJoint"]):
                    joint_info_key = [
                        "x",
                        "y",
                        "z",
                        "depthX",
                        "depthY",
                        "colorX",
                        "colorY",
                        "orientationW",
                        "orientationX",
                        "orientationY",
                        "orientationZ",
                        "trackingState",
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info["jointInfo"].append(joint_info)
                frame_info["bodyInfo"].append(body_info)
            skeleton_sequence["frameInfo"].append(frame_info)

    return skeleton_sequence


def read_xyz(file: str, num_body: int, num_joint: int, max_body: int = 4) -> np.ndarray:
    """
    - Get coordinates x,y,z from .skeleton files
    - Return ndarray with (C,T,V,M) shape.
    - C: num_channel, T: num_frame, V: num_joint, M: num_body
    """
    seq_info = read_skeleton(file)
    data = np.zeros((max_body, seq_info["numFrame"], num_joint, 3))
    for n, f in enumerate(seq_info["frameInfo"]):
        for m, b in enumerate(f["bodyInfo"]):
            for j, v in enumerate(b["jointInfo"]):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v["x"], v["y"], v["z"]]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:num_body]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def get_nonzero_std(s: np.ndarray) -> float:
    """
    Get nonzero std from ndarray with (T,V,C) shape.
    C num_channel, T num_frame, V num_joint
    """
    index = s.sum(-1).sum(-1) != 0  # (T,)
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def gen_joint(
    path_data_input: str,
    path_data_output: str,
    path_data_ignore=None,
    ls_class=None,
    num_frame=300,
    num_joint=25,
    num_body=2,
    max_body=4,
    dict_benchmark=None,
):
    """
    Generate data joint: npy (samples) + pkl (label)
    """
    if path_data_ignore != None:
        with open(path_data_ignore, "r") as f:
            ignored_samples = [line.strip() + ".skeleton" for line in f.readlines()]
    else:
        ignored_samples = []

    for benchmark in dict_benchmark.keys():
        train_joint = []
        train_label = []
        val_joint = []
        val_label = []
        for filename in os.listdir(path_data_input):
            if filename in ignored_samples:
                continue

            extracted_name = analyst.read_meta_data(filename)
            action_class = extracted_name["action_class"]

            if action_class not in ls_class:
                continue

            if analyst.check_benchmark(filename, dict_benchmark[benchmark]):
                train_joint.append(filename)
                train_label.append(action_class)
            else:
                val_joint.append(filename)
                val_label.append(action_class)

        # ============================== Train
        # label
        with open(
            "{}/{}/train/train_label.pkl".format(path_data_output, benchmark),
            "wb",
        ) as f:
            pickle.dump((train_joint, list(train_label)), f)

        # sample
        # fp.shape() = 4091 samples x 3 (x,y,z) x 300 (max frame)  x 25 (num joint)  x 2 (max body true)
        # N x C xT x V x M
        fp = np.zeros(
            (len(train_label), 3, num_frame, num_joint, num_body),
            dtype=np.float32,
        )
        for i, s in enumerate(tqdm(train_joint)):
            data = read_xyz(
                os.path.join(path_data_input, s),
                num_body=num_body,
                num_joint=num_joint,
                max_body=max_body,
            )
            # insert exac number of frames at dimention 2
            fp[i, :, 0 : data.shape[1], :, :] = data
        fp = preprocess.normalize(fp)
        np.save("{}/{}/train/train_joint.npy".format(path_data_output, benchmark), fp)

        # =============================== Validate, the codes are similar to Train part
        with open(
            "{}/{}/val/val_label.pkl".format(path_data_output, benchmark), "wb"
        ) as f:
            pickle.dump((val_joint, list(val_label)), f)
        fp = np.zeros(
            (len(val_label), 3, num_frame, num_joint, num_body), dtype=np.float32
        )
        for i, s in enumerate(tqdm(val_joint)):
            data = read_xyz(
                os.path.join(path_data_input, s),
                num_body=num_body,
                num_joint=num_joint,
                max_body=max_body,
            )
            fp[i, :, 0 : data.shape[1], :, :] = data
        fp = preprocess.normalize(fp)
        np.save("{}/{}/val/val_joint.npy".format(path_data_output, benchmark), fp)


if __name__ == "__main__":
    # create folder to store results
    for b in list(config_glob["benchmarks"].keys()):
        for p in config_glob["phases"]:
            out_path = os.path.join(config_glob["path_data_preprocess"], b, p)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

    gen_joint(
        config_glob["path_data_raw"],
        config_glob["path_data_preprocess"],
        config_glob["path_data_ignore"],
        config_glob["ls_class"],
        config_glob["num_frame"],
        config_glob["num_joint"],
        config_glob["num_body"],
        config_glob["max_body"],
        config_glob["benchmarks"],
    )
