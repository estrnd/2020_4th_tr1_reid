
class StyleFinderConfig(object):
    env = 'default'
    backbone = 'resnet50'
    classify = 'softmax'
    #num_classes = 3039
    # v2
    num_classes = 3226
    num_frames = 6
    num_lens = 7
    num_models = 991

    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    #loss = 'focal_loss'
    loss = 'cross_entropy'
    type = 'style_finder'

    # train_root = '/home/snowyunee/dataset/stylefinder/style_finder_data/stylefinder_v1'
    # train_list = '/home/snowyunee/dataset/stylefinder/style_finder_train.txt'
    # val_list   = '/home/snowyunee/dataset/stylefinder/style_finder_val.txt'
    train_root = None
    train_list = '/home/snowyunee/dataset/stylefinder/vf_train.txt'
    val_list   = '/home/snowyunee/dataset/stylefinder/vf_val.txt'

    # embedding_root = '/home/snowyunee/dataset/stylefinder/style_finder_data/stylefinder_v1'
    # embedding_list = '/home/snowyunee/dataset/stylefinder/embedding_list.txt'

    # embedding_root = '/home/snowyunee/dataset/stylefinder/style_finder_data/v2_crop'
    # embedding_list = '/home/snowyunee/dataset/stylefinder/embedding_list_v2.txt'
    # test_root = '/home/snowyunee/dataset/stylefinder/rf_test_crop'
    # test_list = "/home/snowyunee/dataset/stylefinder/rf_test.txt"

    # testset_1st
    # embedding_root = '/home/snowyunee/dataset/stylefinder/testset_1st_crop/reg'
    # embedding_list = '/home/snowyunee/dataset/stylefinder/embedding_list_testset_1st.txt'
    # test_root = '/home/snowyunee/dataset/stylefinder/testset_1st_crop/query'
    # test_list = "/home/snowyunee/dataset/stylefinder/test_testset_1st.txt"

    # embedding v2 test testset
    embedding_root = '/home/snowyunee/dataset/stylefinder/style_finder_data/v2_crop'
    embedding_list = '/home/snowyunee/dataset/stylefinder/embedding_list_testset_1st_emb_v2_3.txt'
    test_root = '/home/snowyunee/dataset/stylefinder/testset_1st_crop/query'
    test_list = '/home/snowyunee/dataset/stylefinder/test_testset_1st_emb_v2.txt'

    lfw_root = None
    lfw_list = None
    lfw_test_list = None

    # train_batch_size = 64
    # val_batch_size = 16
    # test_batch_size = 16

    train_batch_size = 128
    val_batch_size = 16
    test_batch_size = 16

    input_shape = (3, 224, 224)

    optimizer = 'adam'

    use_gpu = True
    num_workers = 16 # number of worker for loading data
    print_freq = 10
    save_model_freq = 1000

    # checkpoints_path = 'checkpoints_v2_multi'
    # load_model_path = 'resnet50_8.pth'
    # start_epoch = 9
    # freez_layers = [1,2]

    checkpoints_path = 'checkpoints_v2_multi_190312'
    load_model_path = 'resnet50_1.pth'
    start_epoch = 1
    freez_layers = []

    # load_model_path = None
    #test_model_path = 'checkpoints_style_finder_resnet50/resnet50_3.pth'
    # test_model_path = 'weight/Mar05_17-54-31_resnet50_9.pth'
    test_model_path = 'checkpoints_v2_multi/resnet50_26.pth'
    save_interval = 1


    max_epoch = 100
    milestones = [1000]

    lr = 0.00025
    # lr = 0.0005
    #lr = 0.001
    weight_decay = 5e-4

