import yaml
with open("agcn/config/general-config/general_config.yaml", 'r') as f:
    arg = yaml.load(f, Loader=yaml.FullLoader)
import sys
import os 

import numpy as np
from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
from agcn.data_gen import gen_joint_data
from agcn.data_gen import preprocess

import imageio
trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]

def draw_skeleton(input: np.ndarray , output: str):
    data= input
    C, T, V, M = data.shape

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(35, 60)

    images = []
    # show every frame 3d skeleton
    for frame_idx in range(data.shape[1]):

        plt.cla()
        plt.title("Frame: {}".format(frame_idx))

        if output=="raw":
            ax.set_xlim3d([0.5, 2.5])
            ax.set_ylim3d([0, 2])
            ax.set_zlim3d([4, 6])
        else:
            ax.set_xlim3d([-1, 1])
            ax.set_ylim3d([-1, 1])
            ax.set_zlim3d([-1, 1])

        x = data[0, frame_idx, :, 0]
        y = data[1, frame_idx, :, 0]
        z = data[2, frame_idx, :, 0]

        for part in body:
            x_plot = x[part]
            y_plot = y[part]
            z_plot = z[part]
            ax.plot(x_plot, y_plot, z_plot, color='b', marker='o', markerfacecolor='r')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        #create folder
        out_path =  os.path.join(arg['output_visualize'],output,"png")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        plt.savefig(out_path+"/{}.png".format(frame_idx))
        images.append(imageio.imread(out_path+"/{}.png".format(frame_idx)))
        ax.set_facecolor('none')

    imageio.mimsave(out_path+'/../action.gif', images)


if __name__=="__main__":
    falling_down_action= arg["input_data_raw"]+"/S001C001P001R001A043.skeleton"
    
    #draw raw data
    input = gen_joint_data.read_xyz(falling_down_action)
    output="raw"
    draw_skeleton(input, output)

    #draw preprocessed data
    preprocessed_input = np.array(preprocess.pre_normalize(np.expand_dims(input, axis=0)))
    preprocessed_input = np.array(np.squeeze(preprocessed_input, axis=0))
    output="preprocessed"
    draw_skeleton(preprocessed_input, output)
