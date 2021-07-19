import os
import shutil
from enum import IntEnum

import imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from src.main import generator
from src.main.config import cfg_ds
from src.main.feeder.util import change_speed

position_joints =[25,0]
trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]


class SkeletonType(IntEnum):
    RAW = 1
    PREPROCESSED = 2


def draw_skeleton(
    skeleton: np.ndarray,
    type_skeleton: SkeletonType,
    dir_output: str,
    name_gif: str = "action",
    num_joint=26,
    version="tkhar"
) -> None:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(35, 60)

    images = []

    # create folder
    out_path = os.path.join(dir_output, str(type_skeleton.name).lower(), "png")
    os.makedirs(out_path, exist_ok=True)
    for file_img in os.listdir(out_path):
        os.remove(os.path.join(out_path, file_img))

    # show every frame 3d skeleton
    for frame_idx in range(skeleton.shape[1]):

        plt.cla()
        plt.title("Frame: {}".format(frame_idx))

        if type_skeleton == SkeletonType.RAW:
            ax.set_xlim3d([0.5, 2.5])
            ax.set_ylim3d([0, 2])
            ax.set_zlim3d([4, 6])
            x = skeleton[0, frame_idx, :, 0]
            y = skeleton[1, frame_idx, :, 0]
            z = skeleton[2, frame_idx, :, 0]
        elif type_skeleton == SkeletonType.PREPROCESSED:
            if num_joint == 26:body.append(position_joints) 
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
    # shutil.rmtree(out_path, ignore_errors=True)



def visualize(action):
    dir_data = cfg_ds.path_data_raw
    path_data = dir_data + "/%s.skeleton" % (action)

    # draw raw data
    input_raw = generator.joint.read_xyz(path_data)
    zeroes = np.zeros((3, cfg_ds.num_frame, cfg_ds.num_joint, cfg_ds.num_body),dtype=np.float32)
    zeroes[:, :input_raw.shape[1], :input_raw.shape[2], :] = input_raw
    input_raw=zeroes
    draw_skeleton(input_raw, SkeletonType.RAW, cfg_ds.path_visualization, action, num_joint=25)

    # draw preprocessed author - todo

    # draw preprocessed 25 data
    input_preprocess = np.array(generator.processor.normalize(np.expand_dims(input_raw, axis=0), silent=True))[:,:,:,:25,:]
    input_preprocess = np.array(np.squeeze(input_preprocess, axis=0))
    draw_skeleton(input_preprocess, SkeletonType.PREPROCESSED, cfg_ds.path_visualization+"/25 joints", action, num_joint=25)
    
    # # draw preprocessed 26 data
    input_preprocess = np.array(generator.processor.normalize(np.expand_dims(input_raw, axis=0), silent=True))[:,:,:,:26,:]
    input_preprocess = np.array(np.squeeze(input_preprocess, axis=0))
    draw_skeleton(input_preprocess, SkeletonType.PREPROCESSED, cfg_ds.path_visualization+"/26 joints", action, num_joint=26)
    
    #draw preprocessed with speed change
    x0_3 = change_speed(input_preprocess, 0.3) 
    draw_skeleton(x0_3, SkeletonType.PREPROCESSED,cfg_ds.path_visualization, action+"x0.3")




    # draw_skeleton(
    #     generator.get_skeleton_by_frame("output/pose/shaking-hands.mp4"),
    #     SkeletonType.PREPROCESSED,
    #     "./output/",
    #     "abc",
    # )

if __name__ == "__main__":
    ls_class = cfg_ds.ls_class
    ls_action_file=[]

    for cls in ls_class[2:3]:#draw 1 skeleton
        if cls <= 60:
            ls_action_file.append("S001C001P001R001A" + ("00" if cls <10 else "0" if cls <100 else "") + str(cls))
        else:
            ls_action_file.append("S018C001P008R001A" + ("00" if cls <10 else "0" if cls <100 else "") + str(cls))
    
    for action in ls_action_file:
        visualize(action)
