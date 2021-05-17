from .structure import BenchmarkConfig, DatasetConfig

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

config_local = DatasetConfig(
    path_data_raw="/data/thucth/HK202/THESIS/dataset/raw_ntu",
    # path_data_preprocess="/data/thucth/HK202/THESIS/dataset/preprocess2",
    path_data_preprocess="output_genjoint",
    path_data_ignore="/data/thucth/HK202/THESIS/dataset/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    ls_class=[41, 42, 43, 44, 45, 46, 47, 48, 49, 103, 104, 105],
    ls_benmark=[xview, xsub],
    num_body=2,
    num_joint=25,
    num_frame=300,
    max_body=4,
)

config_colab = DatasetConfig(
    path_data_raw="/data/extracts/nturgb+d_skeletons",
    # path_data_preprocess="/data_position_background/preprocess/nturgb+d_skeletons_reorder",
    # path_data_preprocess="/data_zeropadding/preprocess/nturgb+d_skeletons_reorder",
    # path_data_preprocess="/data_preprocess_daily/preprocess/nturgb+d_skeletons_reorder",
    path_data_preprocess="/data_preprocess_daily_26/preprocess/nturgb+d_skeletons_reorder",
    # path_data_preprocess="/data_duppadding/preprocess/nturgb+d_skeletons_reorder",
    path_data_ignore="/data/extracts/samples_with_missing_skeletons.txt",
    path_visualization="output/visualization/",
    ls_class=[41, 42, 43, 44, 45, 46, 47, 48, 49, 103, 104, 105],
    ls_benmark=[xview, xsub],
    num_body=2,
    num_joint=25,
    num_frame=300,
    max_body=4,
)
