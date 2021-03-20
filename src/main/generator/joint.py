import os
import pickle
from typing import Dict, List

import numpy as np
from src.main.config import DatasetConfig, cfg_ds_v1
from tqdm import tqdm
from xcommon import xconsole

from . import processor, util


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


def read_xyz(
    file: str, num_body: int = 2, num_joint: int = 25, max_body: int = 4,
) -> np.ndarray:
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


def gen_joint(name_benchmark: str, ls_filename: List[str], ls_label: List[int], config: DatasetConfig):
    with open("{}/{}_label.pkl".format(config.path_data_preprocess, name_benchmark), "wb") as f:
        pickle.dump((ls_filename, list(ls_label)), f)

    fp = np.zeros(
        (len(ls_label), 3, config.num_frame, config.num_joint, config.num_body), dtype=np.float32,
    )
    for i, s in enumerate(tqdm(ls_filename)):
        data = read_xyz(
            os.path.join(config.path_data_raw, s),
            num_body=config.num_body,
            num_joint=config.num_joint,
            max_body=config.max_body,
        )
        # insert exac number of frames at dimention 2
        fp[i, :, 0: data.shape[1], :, :] = data
    fp = processor.normalize(fp)
    np.save("{}/{}_joint.npy".format(config.path_data_preprocess, name_benchmark), fp)


if __name__ == "__main__":
    config = cfg_ds_v1
    if config.path_data_ignore != None:
        with open(config.path_data_ignore, "r") as f:
            ignored_samples = [line.strip() + ".skeleton" for line in f.readlines()]
    else:
        ignored_samples = []

    for benchmark in config.ls_benmark:
        xconsole.info(benchmark.name)
        train_joint: List[str] = []
        train_label: List[int] = []
        val_joint: List[str] = []
        val_label: List[int] = []
        for filename in os.listdir(config.path_data_raw):
            if filename in ignored_samples:
                continue

            extracted_name: Dict = util.read_meta_data(filename)
            action_class: int = extracted_name["action_class"]

            if action_class not in config.ls_class:
                continue

            if util.check_benchmark(filename, benchmark):
                train_joint.append(filename)
                train_label.append(action_class)
            else:
                val_joint.append(filename)
                val_label.append(action_class)

        gen_joint("train_%s" % (benchmark.name), train_joint, train_label, config)
        gen_joint("val_%s" % (benchmark.name), val_joint, val_label, config)
