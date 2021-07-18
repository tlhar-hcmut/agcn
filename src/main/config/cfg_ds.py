from .structure import BenchmarkConfig, TKHARConfig

xsub = BenchmarkConfig(
    name="xsub",
    setup_number=[],
    camera_id=[],
    performer_id=[
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34,  35,
        38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80,
        81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
    ],
    replication_number=[],
    action_class=[],
)

setup_first60 = [1, 2, 4, 5,  7, 8,  10, 11, 13, 14, 16, 17]
setup_last60 = [18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 32]
xview = BenchmarkConfig(
    name="xview",
    setup_number=setup_first60+setup_last60,
    camera_id=[],
    performer_id=[],
    replication_number=[],
    action_class=[],
)


###################################################################
#                     Sequence                                    #
###################################################################

config_daily_25_sequent_xview = TKHARConfig(
    name       ="config_daily_25_sequent_xview",       
    desc        ="",

    benchmark="xview",

    path_data_raw="/data/extracts/nturgb+d_skeletons",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    path_data_preprocess="/data_preprocess_daily_25/preprocess/nturgb+d_skeletons_reorder",
    ls_benmark=[xview, xsub],
    num_joint=25,

    #common configs
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/config_daily_25_sequent_xview",
    input_size      = (3, 300, 25, 2),
    optim_cfg       ={},
    #configs for temporal stream
    input_size_temporal = (8, 300, 25, 2),
    len_feature_new = [256, 256, 512],
    num_block       =2,
    dropout         =0.2,
    num_head        =8,
)

config_daily_25_sequent_xsub = TKHARConfig(
    name       ="config_daily_25_sequent_xsub",       
    desc        ="",

    benchmark="xsub",

    path_data_raw="/data/extracts/nturgb+d_skeletons",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    path_data_preprocess="/data_preprocess_daily_25/preprocess/nturgb+d_skeletons_reorder",
    ls_benmark=[xview, xsub],
    num_joint=25,

    #common configs
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/config_daily_25_sequent_xsub",
    input_size      = (3, 300, 25, 2),
    optim_cfg       ={},
    #configs for temporal stream
    input_size_temporal = (8, 300, 25, 2),
    len_feature_new = [256, 256, 512],
    num_block       =2,
    dropout         =0.2,
    num_head        =8,
)

config_daily_26_sequent_xview = TKHARConfig(
    name       ="config_daily_26_sequent_xview",       
    desc        ="",

    benchmark="xview",

    path_data_raw="/data/extracts/nturgb+d_skeletons",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    path_data_preprocess="/data_preprocess_daily_26/preprocess/nturgb+d_skeletons_reorder",
    ls_benmark=[xview, xsub],
    num_joint=26,

    #common configs
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/config_daily_26_sequent_xview",
    input_size      = (3, 300, 26, 2),
    optim_cfg       ={},
    #configs for temporal stream
    input_size_temporal = (8, 300, 26, 2),
    len_feature_new = [256, 256, 512],
    num_block       =2,
    dropout         =0.2,
    num_head        =8,
)

config_daily_26_sequent_xsub = TKHARConfig(
    name       ="config_daily_26_sequent_xsub",       
    desc        ="",

    benchmark="xsub",

    path_data_raw="/data/extracts/nturgb+d_skeletons",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    path_data_preprocess="/data_preprocess_daily_26/preprocess/nturgb+d_skeletons_reorder",
    ls_benmark=[xview, xsub],
    num_joint=26,

    #common configs
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/config_daily_26_sequent_xsub",
    input_size      = (3, 300, 26, 2),
    optim_cfg       ={},
    #configs for temporal stream
    input_size_temporal = (8, 300, 26, 2),
    len_feature_new = [256, 256, 512],
    num_block       =2,
    dropout         =0.2,
    num_head        =8,
)


###################################################################
#                     Parallel                                    #
###################################################################
config_daily_25_parallel_xview = TKHARConfig(
    name       ="config_daily_25_parallel_xview",       
    desc        ="",

    benchmark="xview",

    path_data_raw="/data/extracts/nturgb+d_skeletons",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    path_data_preprocess="/data_preprocess_daily_25/preprocess/nturgb+d_skeletons_reorder",
    ls_benmark=[xview, xsub],
    num_joint=25,

    #common configs
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/config_daily_25_parallel_xview",
    input_size      = (3, 300, 25, 2),
    optim_cfg       ={},
    stream          =[0,1],
    
    #configs for temporal stream
    input_size_temporal = (3, 300, 25, 2),
    len_feature_new = [256, 256, 512],
    num_block       =2,
    dropout         =0.2,
    num_head        =8,
)

config_daily_25_parallel_xsub = TKHARConfig(
    name       ="config_daily_25_parallel_xsub",       
    desc        ="",

    benchmark="xsub",

    path_data_raw="/data/extracts/nturgb+d_skeletons",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    path_data_preprocess="/data_preprocess_daily_25/preprocess/nturgb+d_skeletons_reorder",
    ls_benmark=[xview, xsub],
    num_joint=25,

    #common configs
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/config_daily_25_parallel_xsub",
    input_size      = (3, 300, 25, 2),
    optim_cfg       ={},
    stream          =[0,1],
    
    #configs for temporal stream
    input_size_temporal = (3, 300, 25, 2),
    len_feature_new = [256, 256, 512],
    num_block       =2,
    dropout         =0.2,
    num_head        =8,
)

config_daily_26_parallel_xview = TKHARConfig(
    name       ="config_daily_26_parallel_xview",       
    desc        ="",

    benchmark="xview",

    path_data_raw="/data/extracts/nturgb+d_skeletons",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    path_data_preprocess="/data_preprocess_daily_26/preprocess/nturgb+d_skeletons_reorder",
    ls_benmark=[xview, xsub],
    num_joint=26,

    #common configs
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/config_daily_26_parallel_xview",
    input_size      = (3, 300, 26, 2),
    optim_cfg       ={},
    stream          =[0,1],
    
    #configs for temporal stream
    input_size_temporal = (3, 300, 26, 2),
    len_feature_new = [256, 256, 512],
    num_block       =2,
    dropout         =0.2,
    num_head        =8,
)

config_daily_26_parallel_xsub = TKHARConfig(
    name       ="config_daily_26_parallel_xsub",       
    desc        ="",

    benchmark="xsub",

    path_data_raw="/data/extracts/nturgb+d_skeletons",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    path_data_preprocess="/data_preprocess_daily_26/preprocess/nturgb+d_skeletons_reorder",
    ls_benmark=[xview, xsub],
    num_joint=26,

    #common configs
    output_train    = "/content/gdrive/Shareddrives/Thesis/result_train/config_daily_26_parallel_xsub",
    input_size      = (3, 300, 26, 2),
    optim_cfg       ={},
    stream          =[0,1],
    
    #configs for temporal stream
    input_size_temporal = (3, 300, 26, 2),
    len_feature_new = [256, 256, 512],
    num_block       =2,
    dropout         =0.2,
    num_head        =8,
)



###################################################################
#                     Local -for analysis                         #
###################################################################

config_local_xview = TKHARConfig(
    name       ="config_local_xview",       
    desc        ="",

    benchmark="xview",

    path_data_raw="/data/thucth/HK202/THESIS/dataset/raw_ntu",
    path_data_ignore=None,
    path_visualization=None,
    path_data_preprocess="/data/thucth/HK202/THESIS/dataset/daily_26",
    ls_benmark=[xview, xsub],
    num_joint=25,

    #common configs
    output_train    = None,
    input_size      =None,
    optim_cfg       =None,
    #configs for temporal stream
    input_size_temporal =None,
    len_feature_new =None,
    num_block       =None,
    dropout         =None,
    num_head        =None,
)

config_local_xsub = TKHARConfig(
    name       ="config_local_xsub",       
    desc        ="",

    benchmark="xsub",

    path_data_raw="/data/thucth/HK202/THESIS/dataset/raw_ntu",
    path_data_ignore=None,
    path_visualization=None,
    path_data_preprocess="/data/thucth/HK202/THESIS/dataset/daily_26",
    ls_benmark=[xview, xsub],
    num_joint=25,

    #common configs
    output_train    = None,
    input_size      = None,
    optim_cfg       = None,
    #configs for temporal stream
    input_size_temporal = None,
    len_feature_new = None,
    num_block       = None,
    dropout         = None,
    num_head        = None,
)



###################################################################
#                     Local -for debug                            #
###################################################################
config_local_xview_debug = TKHARConfig(
    name       ="config_local_xview",       
    desc        ="",

    benchmark="xview",

    path_data_raw="/data/extracts/nturgb+d_skeletons",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    path_data_preprocess="/data_preprocess_daily_26/preprocess/nturgb+d_skeletons_reorder",
    ls_benmark=[xview, xsub],
    num_joint=26,

    #common configs
    output_train    = "output_train",
    input_size      = (3, 300, 26, 2),
    optim_cfg       ={},
    #configs for temporal stream
    input_size_temporal = (8, 300, 26, 2),
    len_feature_new = [256, 256, 512],
    num_block       =3,
    dropout         =0.2,
    num_head        =8,
    batch_size=25,
)