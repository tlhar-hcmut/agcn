
class XSub:
    performer_id = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]

class XView:
    setup_number  = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
    # training_cameras = [2, 3]

class BenchMark:
    xsub=XSub()
    xview=XView()

class GeneralConfig:
    input_data_raw  = "/data/extracts/nturgb+d_skeletons"

    ignored_sample_path = '/data/extracts/NTU_RGBD120_samples_with_missing_skeletons.txt'

    output_data_preprocess = "agcn/output_data_preprocess_full_test"

    chosen_class = [41,42,43,44,45,46,47,48,49,103,104,105]

    # training set for each benchmark
    benchmarks= BenchMark()

    part            = ['train', 'val']

    max_body_true   = 2
    max_body_kinect = 4
    num_joint       = 25
    max_frame       = 300


    #PARAMETERS FOR VISUALIZE
    output_visualize=  "agcn/output_visualize/"
