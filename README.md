# MS-AGCN
Multi-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition.

# Data Preparation

- Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D). Then put it under the data directory:

```
-data\
  -nturgbd_raw\  
    -nturgb+d_skeletons\
      ...
    -samples_with_missing_skeletons.txt
```

[https://github.com/shahroudy/NTURGB-D]: NTU-RGB+D


- Download skeleton data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) by command-line:

```
pip3 install gdown # or pip
gdown https://drive.google.com/uc?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H
gdown https://drive.google.com/uc?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w
unzip nturgbd_skeletons_s001_to_s017.zip -d data/nturgbd_raw/nturgb+d_skeletons
unzip nturgbd_skeletons_s018_to_s032.zip -d data/nturgbd_raw/nturgb+d_skeletons
```

- Download pretrain model from [lshiwjx](https://github.com/lshiwjx)

```
wget https://github.com/lshiwjx/2s-AGCN/releases/download/v0.0/model.zip
unzip model -d ./runs
```
## Preprocess data:

* Note: config parameters in `config/general-config/general_config.yaml` before gen data.

- generate joint data:          `python3 -m 2s-agcn.data_gen.gen_joint_data`

- generate bone data:           `python3 -m 2s-agcn.data_gen.gen_bone_data`
    
- generate motion data:          `python3 -m 2s-agcn.data_gen.gen_motion_data`

- and the same for others.
# Training & Testing

Change the config file depending on what you want.


  `python3 main.py --config ./config/nturgbd-cross-view/train_joint.yaml`

  `python3 main.py --config ./config/nturgbd-cross-view/train_bone.yaml`

To ensemble the results of joints and bones, run test firstly to generate the scores of the softmax layer. 

  `python3 main.py --config ./config/nturgbd-cross-view/test_joint.yaml`

  `python3 main.py --config ./config/nturgbd-cross-view/test_bone.yaml`

Then combine the generated scores with: 

  `python3 ensemble.py` --datasets ntu/xview
