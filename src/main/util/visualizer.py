import os
from enum import IntEnum

import imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]


class SkeletonType(IntEnum):
    RAW = 1
    PREPROCESSED = 2


def draw_skeleton(skeleton: np.ndarray, type: SkeletonType, dir_output: str, name_gif: str = "action") -> None:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(35, 60)

    images = []

    # create folder
    out_path = os.path.join(dir_output, str(type.name).lower(), "png")
    os.makedirs(out_path, exist_ok=True)
    for file_img in os.listdir(out_path):
        os.remove(os.path.join(out_path, file_img))

    # show every frame 3d skeleton
    for frame_idx in range(skeleton.shape[1]):

        plt.cla()
        plt.title("Frame: {}".format(frame_idx))

        if type == SkeletonType.RAW:
            ax.set_xlim3d([0.5, 2.5])
            ax.set_ylim3d([0, 2])
            ax.set_zlim3d([4, 6])
        elif type == SkeletonType.PREPROCESSED:
            ax.set_xlim3d([-1, 1])
            ax.set_ylim3d([-1, 1])
            ax.set_zlim3d([-1, 1])

        x = skeleton[0, frame_idx, :, 0]
        y = skeleton[1, frame_idx, :, 0]
        z = skeleton[2, frame_idx, :, 0]

        for part in body:
            x_plot = x[part]
            y_plot = y[part]
            z_plot = z[part]
            ax.plot(x_plot, y_plot, z_plot, color="b", marker="o", markerfacecolor="r")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.savefig(out_path + "/{}.png".format(frame_idx))
        images.append(imageio.imread(out_path + "/{}.png".format(frame_idx)))
        ax.set_facecolor("none")

    imageio.mimsave(out_path + "/../%s.gif" % name_gif, images)
