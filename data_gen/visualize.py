import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from data_gen.gen_joint_data import read_xyz
import imageio
trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]

if __name__ == "__main__":
    data: np.ndarray = read_xyz("/data/extracts/nturgbd_raw/nturgb+d_skeletons/S017C003P020R002A060.skeleton")
    C, T, V, M = data.shape

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(20, -45)

    images = []
    # show every frame 3d skeleton
    for frame_idx in range(data.shape[1]):
        plt.cla()
        plt.title("Frame: {}".format(frame_idx))

        ax.set_xlim3d([0.5, 1.5])
        ax.set_ylim3d([4, 8])
        ax.set_zlim3d([0, 1])

        x = data[0, frame_idx, :, 0]
        y = data[1, frame_idx, :, 0]
        z = data[2, frame_idx, :, 0]

        for part in body:
            x_plot = x[part]
            y_plot = y[part]
            z_plot = z[part]
            ax.plot(x_plot, z_plot, y_plot, color='b', marker='o', markerfacecolor='r')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        plt.savefig("dist/{}.png".format(frame_idx))
        images.append(imageio.imread("dist/{}.png".format(frame_idx)))
        ax.set_facecolor('none')

    imageio.mimsave('dist/action.gif', images)
            