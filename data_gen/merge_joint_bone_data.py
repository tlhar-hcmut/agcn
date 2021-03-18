import yaml
with open("agcn/config/general-config/general_config.yaml", 'r') as f:
    default_arg = yaml.load(f, Loader=yaml.FullLoader)
import os
import numpy as np

phases = {
    'train', 'val'
}

datasets = {
    'xview', 'xsub'
}


def gen_joint_motion(dataset, phase):
    '''
    Merge types of data. Run this after gen joint data, gen bone data.
    '''
    print(dataset, phase)
    file_joint = default_arg['output_data_preprocess']+'/%s/%s/%s_joint.npy' % (dataset, phase, phase)
    file_bone = default_arg['output_data_preprocess']+'/%s/%s/%s_bone.npy' % (dataset, phase, phase)
    file_joint_bone = default_arg['output_data_preprocess']+'/%s/%s/%s_joint_bone.npy' % (dataset, phase, phase)
    data_jpt = np.load(file_joint)
    data_bone = np.load(file_bone)
    data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)

    np.save(file_joint_bone, data_jpt_bone)


if __name__ == "__main__":
    for dataset in datasets:
        for phase in phases:
            gen_joint_motion(dataset, phase)
