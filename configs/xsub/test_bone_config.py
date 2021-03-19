from agcn.configs.common_configs import *

class ConfigTestBoneLoader:
    
    
    # feeder
    feeder="feeders.feeder.Feeder"
    train_feeder_args= TestFeederArgs(
                                        data_path="/data/ntu/xsub/val_data_bone.npy",
                                        label_path="/data/ntu/xsub/val_label.pkl",
                                        debug=False,
                                    )
  
    # model
    model="model.agcn.UnitAGCN"
    model_args= ModelArgs(
                            num_class= 60,
                            num_point= 25,
                            num_person= 2,
                            graph= "graph.ntu_rgb_d.Graph",
                            graph_args=GraphArgs(labeling_mode= 'spatial')
                        )

    # training
    phase="test"
    device=[0]
    batch_size=6
    test_batch_size=64
    weights= "./runs/ntu_cs_agcn_bone-49-31300.pt"

    work_dir="./work_dir/ntu/xsub/agcn_test_bone"
    model_saved_name="./runs/ntu_cs_agcn_test_bone"
    save_score= True
