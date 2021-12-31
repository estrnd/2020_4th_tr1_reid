
class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 10575
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    Finetune = False

    train_root = '/home/snowyunee/dataset/CASIA-maxpy-clean_crop2'
    train_list = '/home/snowyunee/dataset/casia2_train.txt'
    val_list = '/home/snowyunee/dataset/casia2_val.txt'

    test_root = '/home/snowyunee/dataset/lfw_crop'
    test_list = '/home/snowyunee/dataset/lfw_test_pair_new.txt'

    lfw_root = '/home/snowyunee/dataset/lfw_crop'
    lfw_list = '/home/snowyunee/dataset/lfw_total.txt'
    lfw_test_list = '/home/snowyunee/dataset/lfw_test_pair_new.txt'

    checkpoints_path = 'checkpoints'
    start_epoch = 0
    load_model_path = 'checkpoints/resnet18_2.pth'
    #load_model_path = None
    test_model_path = 'checkpoints/resnet18_50.pth'
    save_interval = 2

    train_batch_size = 128
    test_batch_size = 128

    input_shape = (3, 224, 224)

    #optimizer = 'sgd'
    optimizer = 'adam'

    use_gpu = True
    num_workers = 16 # number of worker for loading data
    print_freq = 100
    save_model_freq = 1000

    max_epoch = 200
    lr = 1e-2
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4






    
