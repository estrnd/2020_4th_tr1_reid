
class ReIDConfig(object):
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

    train_root = '/data/kts123/aihub/reid'
    train_list = '/data/kts123/aihub/reid/img_list_train.txt'
    
    val_root = '/data/kts123/aihub/reid'
    val_list = '/data/kts123/aihub/reid/img_list_val.txt'

    embedding_root = '/data/kts123/aihub/reid'
    embedding_list = '/data/kts123/aihub/reid/img_list_val.txt'
    
    test_root = '/data/kts123/aihub/reid'
    test_list = '/data/kts123/aihub/reid/img_list_val.txt'

    
    lfw_root = None
    lfw_list = None
    lfw_test_list = None
    
    
    train_batch_size = 128
    val_batch_size = 16
    test_batch_size = 16

    input_shape = (3, 224, 224)

    optimizer = 'adam'

    use_gpu = True
    num_workers = 16 # number of worker for loading data
    print_freq = 10
    save_model_freq = 1000

    
    checkpoints_path = 'checkpoints_res50_base'
    
    start_epoch = 1
    freez_layers = []

    load_model_path = None
    #test_model_path = 'checkpoints_style_finder_resnet50/resnet50_3.pth'
    # test_model_path = 'weight/Mar05_17-54-31_resnet50_9.pth'
    #test_model_path = 'checkpoints_v2_multi/resnet50_26.pth'
    save_interval = 1


    max_epoch = 100
    milestones = [1000]

    lr = 0.00025
    # lr = 0.0005
    #lr = 0.001
    weight_decay = 5e-4

