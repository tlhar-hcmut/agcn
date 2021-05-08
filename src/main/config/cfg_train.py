output_train = "/content/gdrive/Shareddrives/Thesis/result_train/temporal_stream/batch_aggrigate/26_joints_75F_150Frame/first"
# output_train    =   "output/train/temporal"
batch_size      =   8
stream          =   [1]
input_size =(3, 150, 26, 2) 
#temporal
len_feature_new = [78, 78, 128, 128, 128]
num_block = 3
dropout = 0.2
num_head = 5
