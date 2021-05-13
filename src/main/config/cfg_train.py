from dataclasses import dataclass
from typing import Any

@dataclass
class CfgTrain:
        name            :str
        output_train    :str
        stream          :list
        input_size      :tuple
        desc            :str
        len_feature_new :int
        num_block       :int
        dropout         :float
        num_head        :int
        optim           :str
        loss            :str    
        batch_size      :int = 8
        pretrained_path :str = None
        num_of_epoch    :int = 200
        num_class       :int = 12

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
    desc            =   '''
                            This has good performance: 
                                2021-05-12 16:18:14,167 INFO -  epoch: 19   loss: 0.82846   acc: 0.71506    ----------BEST
                        ''',
    output_train    = "output_multihead/local_1",
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
    output_train    = "output_26channel1/local_2",
    stream          =[1],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [26, 26, 64],
    num_block       =3,
    dropout         =0.2,
    num_head        =5,
    optim           ="adam",
    loss            ="crossentropy"
)

cfgTrainRemote = CfgTrain(
    
    name            = "test_train_local",
    desc            = "some thing",
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/26_joints/first",
    stream          =[1],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [26, 26, 64],
    num_block       =3,
    dropout         =0.5,
    num_head        =4,
    optim           ="adam",
    loss            ="crossentropy"
)

cfgTrainLocalMultihead = CfgTrain(
    
    name            = "local_multihead",
    desc            =  '''
                        stream          =[2],
                        input_size      = (3, 300, 26, 2),
                        
                        len_feature_new = [32, 32, 64, 64],
                        num_block       =4,
                        dropout         =0.2,
                        num_head        =10,
                        optim           ="adam",
                        loss            ="crossentropy"
                        ''',
    output_train    = "output_multihead12/local_multihead1",
    stream          =[2],
    input_size      = (3, 300, 26, 2),
   
    len_feature_new = [32, 32, 64, 64],
    num_block       =3,
    dropout         =0.2,
    num_head        =5,
    optim           ="adam",
    loss            ="crossentropy",
    num_class       =12
)

cfgTrainLocalMultihead1 = CfgTrain(
    
    name            = "multiple2",
    desc            =  '''
                        stream          =[2],
                        input_size      = (3, 300, 26, 2),
                        
                        len_feature_new = [64, 64, 64, 64],
                        num_block       =4,
                        dropout         =0.2,
                        num_head        =8,
                        optim           ="adam",
                        loss            ="crossentropy",
                        num_class       =12
                        ''',
    output_train    = "output_multiple/multiple2",
    stream          =[2],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [64, 64, 64, 64],
    num_block       =4,
    dropout         =0.2,
    num_head        =8,
    optim           ="adam",
    loss            ="crossentropy",
    num_class       =12

)

cfgTrainLocalMultihead2 = CfgTrain(
    
    name            = "multiple1",
    desc            =  '''
                        output_train    = "output_multiple_new_6/multiple1",
                        stream          =[2],
                        input_size      = (3, 300, 26, 2),
                        
                        len_feature_new = [32, 32, 64, 64, 64, 64],
                        num_block       =6,
                        dropout         =0.2,
                        num_head        =8,
                        optim           ="adam",
                        loss            ="crossentropy",
                        num_class       =12
                        ''',
    output_train    = "output_multiple_new_6/multiple1",
    stream          =[2],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [32, 32, 64, 64, 32, 32],
    num_block       =6,
    dropout         =0.2,
    num_head        =8,
    optim           ="adam",
    loss            ="crossentropy",
    num_class       =12

)
