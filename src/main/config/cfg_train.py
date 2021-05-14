from dataclasses import dataclass
from typing import Any

@dataclass
class CfgTrain:
        name            :str
        desc            :str
        output_train    :str
        stream          :list
        input_size      :tuple
        
        len_feature_new :int
        num_block       :int
        dropout         :float
        num_head        :int
        optim           :str
        loss            :str    
        batch_size      :int = 116
        pretrained_path :str = None
        num_of_epoch    :int = 200

# class CfgTrain1(CfgTrain):
#     name            = "test_train_1"
#     desc            = "some thing"
#     output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/26_joints/first"
#     stream          =[1]
#     input_size      = (3, 300, 26, 2)
    
#     len_feature_new = [26, 26, 64]
#     num_block       =3
#     dropout         =0.2
#     num_head        =5
#     optim           ="adam"
#     loss            ="crossentropy"

cfgTrainLocal = CfgTrain(
    
    name            = "local_1",
    desc            = "some thing",
    output_train    = "output/local_1",
    stream          =[1],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [26, 26, 64],
    num_block       =3,
    dropout         =0.2,
    num_head        =5,
    optim           ="adam",
    loss            ="crossentropy"
)

cfgTrainLocal1 = CfgTrain(
    
    name            = "local_2",
    desc            = "some thing",
    output_train    = "output/local_2",
    stream          =[1],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [26, 26, 25, 64],
    num_block       =4,
    dropout         =0.5,
    num_head        =4,
    optim           ="adam",
    loss            ="crossentropy"
)

cfgTrainRemote = CfgTrain(
    
    name            = "test_train_local",
    desc            = "some thing",
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/khoidd/ihateyou",
    stream          =[1],
    input_size      = (3, 300, 25, 2),
    
    len_feature_new = [25, 25, 64],
    num_block       =3,
    dropout         =0.5,
    num_head        =4,
    optim           ="adam",
    loss            ="crossentropy"
)