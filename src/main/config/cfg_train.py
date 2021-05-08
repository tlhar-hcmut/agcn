class CfgTrain:
    output_train    =None
    batch_size      =8 #the same batchsize for all model
    stream          =None
    input_size      =None
    
    len_feature_new =None
    num_block       =None
    dropout         =None
    num_head        =None
    optim           =None
    pretrained_path =None

class CfgTrain1(CfgTrain):
    desc            = "test_train_1"
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/26_joints/first"
    stream          =[1]
    input_size      = (3, 300, 26, 2)
    
    len_feature_new = [26, 26, 64]
    num_block       =3
    dropout         =0.2
    num_head        =5
    optim           ="adam"

class CfgTrainLocal(CfgTrain):
    desc            = "test_train_1"
    output_train    = "output/test"
    stream          =[1]
    input_size      = (3, 300, 26, 2)
    
    len_feature_new = [26, 26, 64]
    num_block       =3
    dropout         =0.2
    num_head        =5
    optim           ="adam"