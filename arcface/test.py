from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    data_list = []
    for pair in pairs:
        splits = pair.split()
        if splits[0] not in data_list:
            data_list.append(splits[0])
        if splits[1] not in data_list:
            data_list.append(splits[1])

    return data_list


def load_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    image = cv2.resize(image, (224, 224))
    #image = image.transpose((0,3,1,2))
   # 2,0,1))

    image2 = np.fliplr(image)
    image = image[np.newaxis, :, :, :]
    #print('image shape:', image.shape)
    image2 = image2[np.newaxis, :, :, :]
    image = np.concatenate((image, image2), axis=0)
    #print('concatenate image shape:', image.shape)
    image = image.astype(np.float32, copy=False)
    #print('image shape:', image.shape)
    image /= 255
    image -= MEAN
    image /= STD
    # image -= 127.5
    # image /= 127.5
    image = image.transpose((0,3,1,2))
    #print('transpose image shape:', image.shape)
    return image


# def get_features(model, test_list, batch_size=10):
#     features = None
#     cnt = 0
#     train_dataset = DataSet(opt.lfw_root, opt.lfw_list, phase='test',
#                             input_shape=opt.input_shape)
#     loader = data.DataLoader(train_dataset,
#                              batch_size=opt.test_batch_size,
#                              shuffle=False,
#                              num_workers=opt.num_workers)
#     total_num = len(loader)
#     for i, (imgs, labels) in tqdm(enumerate(loader), total=total_num) :
# 
#         if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
#             cnt += 1
# 
#             output = model(data)
#             output = output.data.cpu().numpy()
#             feature = output
# 
#             print('get_features: ', feature.shape)
# 
#             if features is None:
#                 features = feature
#             else:
#                 features = np.vstack((features, feature))
# 
# 
#     return features, cnt


def get_features(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    print('get_features, len:', len(test_list))
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))
            continue

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
    no_model_keys = [k for k, _ in pretrained_dict.items() if k in model_dict]
    print('no_model_keys:', no_model_keys)

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]

    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score > th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_features(model, img_paths, batch_size=batch_size)
    t = time.time() - s
    print('totla time is {}, average time is {}'.format(t, t/cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshhold: ', th)
    return acc


def main():
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = DataParallel(model)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device('cuda'))

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]
    img_paths = img_paths[:5]

    model.eval()
    lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)


if __name__ == '__main__':
    main()


