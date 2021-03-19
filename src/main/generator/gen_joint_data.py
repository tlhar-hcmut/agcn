# load config
import yaml

with open("agcn/config/general-config/general_config.yaml", "r") as f:
    arg = yaml.load(f, Loader=yaml.FullLoader)


import os
import numpy as np
import pickle
from tqdm import tqdm

from agcn.data_gen.preprocess import pre_normalize
from agcn.utils import utils

chosen_class = arg["chosen_class"]
max_body_true = arg["max_body_true"]
max_body_kinect = arg["max_body_kinect"]
num_joint = arg["num_joint"]
max_frame = arg["max_frame"]

benchmarks = arg["benchmarks"]
parts = arg["phases"]


def read_skeleton_filter(file):
    """
    put all infos of .skeleton files  into dictionary "skeleton_sequence"
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


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):
    """
    Get coordinates x,y,z from .skeleton files
    """
    seq_info = read_skeleton_filter(file)
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
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gen_joint(
    input_data_raw,
    path_data_preprocess,
    ignored_sample_path=None,
    chosen_class=None,
    benchmark=None,
    part=None,
):
    """
    Generate data joint: npy (samples) + pkl (label)
    """
    if ignored_sample_path != None:
        with open(ignored_sample_path, "r") as f:
            ignored_samples = [line.strip() + ".skeleton" for line in f.readlines()]
    else:
        ignored_samples = []

    for benchmark_ in benchmark.keys():
        train_joint = []
        train_label = []
        val_joint = []
        val_label = []
        for filename in os.listdir(input_data_raw):
            if filename in ignored_samples:
                continue

            extracted_name = utils.read_name(filename)
            action_class = extracted_name["action_class"]

            if action_class not in chosen_class:
                continue

            if utils.checkBenchmark(benchmark=benchmark_, filename=filename):
                train_joint.append(filename)
                train_label.append(action_class)
            else:
                val_joint.append(filename)
                val_label.append(action_class)

        # ============================== Train
        # label
        with open(
            "{}/{}/train/train_label.pkl".format(path_data_preprocess, benchmark_),
            "wb",
        ) as f:
            pickle.dump((train_joint, list(train_label)), f)

        # sample
        # fp.shape() = 4091 samples x 3 (x,y,z) x 300 (max frame)  x 25 (num joint)  x 2 (max body true)
        # N x C xT x V x M
        fp = np.zeros(
            (len(train_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32
        )
        for i, s in enumerate(tqdm(train_joint)):
            data = read_xyz(
                os.path.join(input_data_raw, s),
                max_body=max_body_kinect,
                num_joint=num_joint,
            )
            # insert exac number of frames at dimention 2
            fp[i, :, 0 : data.shape[1], :, :] = data
        fp = pre_normalize(fp)
        np.save(
            "{}/{}/train/train_joint.npy".format(path_data_preprocess, benchmark_), fp
        )

        # =============================== Validate, the codes are similar to Train part
        with open(
            "{}/{}/val/val_label.pkl".format(path_data_preprocess, benchmark_), "wb"
        ) as f:
            pickle.dump((val_joint, list(val_label)), f)
        fp = np.zeros(
            (len(val_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32
        )
        for i, s in enumerate(tqdm(val_joint)):
            data = read_xyz(
                os.path.join(input_data_raw, s),
                max_body=max_body_kinect,
                num_joint=num_joint,
            )
            fp[i, :, 0 : data.shape[1], :, :] = data
        fp = pre_normalize(fp)
        np.save("{}/{}/val/val_joint.npy".format(path_data_preprocess, benchmark_), fp)


if __name__ == "__main__":
    # create folder to store results
    for b in list(benchmarks.keys()):
        for p in parts:
            out_path = os.path.join(arg["path_data_preprocess"], b, p)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

    gen_joint(
        arg["input_data_raw"],
        arg["path_data_preprocess"],
        arg["ignored_sample_path"],
        chosen_class,
        benchmarks,
        parts,
    )
