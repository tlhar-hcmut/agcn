from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class CfgTrain:
        name            :str
        output_train    :str
        input_size      :tuple
        desc            :str
        len_feature_new :int
        num_block       :int
        dropout         :float
        num_head        :int
        optim           :str
        loss            :str    
        stream          :list = None
        input_size_temporal: tuple =None
        optim_cfg       :Dict[str, object] = field(default_factory=lambda: {}) #to avoid use the same dictionary (immutable) for all objects
        batch_size      :int = 8
        pretrained_path :str = None
        num_of_epoch    :int = 200
        num_class       :int = 12


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
    
    name            = "adam",
    desc            =  '''
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/adam",
    stream          =[2],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [64, 64, 64, 64, 64, 64],
    num_block       =4,
    dropout         =0.2,
    num_head        =8,
    optim           ="adam",
    loss            ="crossentropy",
    num_class       =12
                        ''',
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/adam",
    stream          =[2],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [64, 64, 64, 64, 64, 64],
    num_block       =4,
    dropout         =0.2,
    num_head        =8,
    optim           ="adam",
    optim_cfg       ={},
    loss            ="crossentropy",
    num_class       =12

)


cfgTrainLocalMultihead2 = CfgTrain(
    
    name            = "sgd",
    desc            =  '''
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/sgd",
    stream          =[2],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [64, 64, 64, 64, 64, 64],
    num_block       =4,
    dropout         =0.2,
    num_head        =8,
    optim           ="adam",
    loss            ="crossentropy",
    num_class       =12
                        ''',
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/sgd",
    stream          =[2],
    input_size      = (3, 300, 26, 2),
    
    len_feature_new = [64, 64, 64, 64, 64, 64],
    num_block       =4,
    dropout         =0.2,
    num_head        =8,
    optim           ="sgd",
    optim_cfg       ={"lr": 0.01},
    loss            ="crossentropy",
    num_class       =12

)


cfgTrainSequential1 = CfgTrain(
    
    name            = "first",
    desc            =  '''
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/sequential/first",
    # output_train    = "output_sequential",
    stream          =None,
    input_size      = (3, 300, 25, 2),
    
    len_feature_new = [64, 64, 64, 64, 64, 64],
    num_block       =4,
    dropout         =0.2,
    num_head        =8,
    optim           ="adam",
    optim_cfg       ={},
    loss            ="crossentropy",
    num_class       =12
                        ''',
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/sequential/first",
    # output_train    = "output_sequential",
    stream          =None,
    input_size      = (3, 300, 25, 2),
    
    len_feature_new = [64, 64, 64, 64, 64, 64],
    num_block       =4,
    dropout         =0.2,
    num_head        =8,
    optim           ="adam",
    optim_cfg       ={},
    loss            ="crossentropy",
    num_class       =12
)

cfgTrainSequential2 = CfgTrain(
    
    name            = "second",
    desc            =  '''
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/sequential/second",
    # output_train    = "output_sequential",
    stream          =None,
    input_size      = (3, 300, 25, 2),
    
    len_feature_new = [64, 64, 64, 64, 64, 64],
    num_block       =4,
    dropout         =0.2,
    num_head        =8,
    optim           ="adam",
    optim_cfg       ={},
    loss            ="crossentropy",
    num_class       =12
                        ''',
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/sequential/second",
    # output_train    = "output_sequential",
    stream          =None,
    input_size      = (3, 300, 25, 2),
    
    len_feature_new = [64, 64, 64, 64, 64, 64],
    num_block       =4,
    dropout         =0.2,
    num_head        =8,
    optim           ="adam",
    optim_cfg       ={},
    loss            ="crossentropy",
    num_class       =12
)

cfgTrainSequential3 = CfgTrain(
    
    name            = "eighth_backup",
    desc            =  '''
    #common configs
    output_train    = "output_sequential_3",
    input_size      = (3, 300, 25, 2),
    optim           ="adam", #adam or sgd
    optim_cfg       ={},
    loss            ="crossentropy",
    
    #configs for temporal stream
    input_size_temporal      = (16, 300, 25, 2),
    len_feature_new = [16, 16, 32, 32, 16, 16],
    num_block       =6,
    dropout         =0.2,
    num_head        =4,
    num_class       =12

    #configs for spatial stream

    giam chieu channel ve 16
    giam joint ve 5
    tang do sau 6 unit
    giam len_feature_new 
                        ''',
    #common configs
    # output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/sequential/eighth_backup",
    output_train    = "output_sequential_4",
    input_size      = (3, 300, 25, 2),
    optim           ="adam", #adam or sgd
    optim_cfg       ={},
    loss            ="crossentropy",
    
    #configs for temporal stream
    input_size_temporal      = (16, 300, 25, 2),
    len_feature_new = [16, 16, 32, 32, 16, 16],
    num_block       =6,
    dropout         =0.2,
    num_head        =4,
    num_class       =12

    #configs for spatial stream

)


cfgTrainSequential5 = CfgTrain(
    
    name            = "eighth_backup",
    desc            =  '''
    #common configs
    output_train    = "output_sequential_5",
    input_size      = (3, 300, 25, 2),
    optim           ="adam", #adam or sgd
    optim_cfg       ={},
    loss            ="crossentropy",
    
    #configs for temporal stream
    input_size_temporal      = (16, 300, 25, 2),
    len_feature_new = [16, 16, 32, 32, 16, 16],
    num_block       =6,
    dropout         =0.2,
    num_head        =4,
    num_class       =12

    #configs for spatial stream

    giam chieu channel ve 16
    giu nguyen joint
    mean joint, mean C_new
                        ''',
    #common configs
    # output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/update_0514/sequential/eighth_backup",
    output_train    = "output_sequential_5",
    input_size      = (3, 300, 25, 2),
    optim           ="adam", #adam or sgd
    optim_cfg       ={},
    loss            ="crossentropy",
    
    #configs for temporal stream
    input_size_temporal      = (16, 300, 25, 2),
    len_feature_new = [16, 16, 32, 32, 16, 16],
    num_block       =6,
    dropout         =0.2,
    num_head        =4,
    num_class       =12

    #configs for spatial stream

)
