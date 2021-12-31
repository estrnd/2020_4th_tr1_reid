import os

from data import get_datasets, get_loaders
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import itertools

import pandas as pd
import time
import cv2
import PIL
import numpy as np
import random
from tqdm import tqdm

from models import *
from config_style_finder import StyleFinderConfig
# from config_style_finder_res18 import StyleFinderConfig

import glob


def load_image(img_path, input_shape):
    import matplotlib.pyplot as plt

    try:
        image = PIL.Image.open(img_path)
    except:
        print('\n!!!!!!!!!!!! no image', img_path)
        return None

    image_scale = int(input_shape[1] * 1.125)
    image = image.resize((image_scale, image_scale))
    x = int((image_scale-input_shape[1])/2)
    image = image.crop((x,x,x+input_shape[1],x+input_shape[2]))
    image = np.array(image)
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    image = image[np.newaxis, :, :, :]
    image = image.astype(np.float32, copy=False)
    image /= 255
    image -= MEAN
    image /= STD
    image = image.transpose((0,3,1,2))
    # print('\n------------', image.shape)
    return image


def get_list(f_name, root):
    df = pd.read_csv(f_name)
    ls = df.to_dict('records')
    return [(os.path.join(root, v['path']),v['label']) for v in ls]


# def cosin_metric(x, embeddings):
#     x_norm = np.linalg.norm(x)
#     embeddings_norm = np.linalg.norm(embeddings, 0)
#     assert(embeddings_norm.shape[0] == embeddings.shape[0])
#     norms = x_norm * embeddings_norm
#     embeddings = np.transpose(embeddings)
#     return np.dot(x, embeddings) / nors

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def get_feature_dict(ids, features):
    # reatures : [{'query':img_path, 'label': label}]
    fe_dict = {}
    for i, v in enumerate(ids):
        # print(i, v)
        if v['label'] not in fe_dict:
            fe_dict[v['label']] = {'dir': v['dir'], 'features':[]}
        fe_dict[v['label']]['features'].append(features[i])

    # normalize
    for i, k in enumerate(fe_dict):
        ls = fe_dict[k]['features']
        ls = [e / np.linalg.norm(e) for e in ls]
        fe_dict[k]['features'] = ls
        # print(' average:', fe_dict[k].shape)

    # average features
    for i, k in enumerate(fe_dict):
        fe_dict[k]['features'] = np.average(np.array(fe_dict[k]['features']), axis=0)
        # print(' average:', fe_dict[k].shape)

    return fe_dict


def get_features1(model, embedding_list, batch_size, input_shape):
    images = None
    features = None

    embedded_list = []
    #print('get_features, len:', len(embedding_list))
    s = time.time()
    for i, (img_path, label) in enumerate(embedding_list):
        image = load_image(img_path, input_shape)
        if image is None:
            print('read {} error'.format(img_path))
            continue

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        embedded_list.append({'query':img_path,
            'label': label,
            'dir': img_path.split('/')[-2]
            })
        #print('i:', i, ', label:', label, ', path:', img_path)

        if images.shape[0] % batch_size == 0 or i == len(embedding_list) - 1:
            # print('image:', images.shape)
            # print('ids: ', ids)

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            outputs = model(data)
            outputs = outputs.data.cpu().numpy()

            if features is None:
                features = outputs
            else:
                features = np.vstack((features, outputs))

            images = None

    assert(len(embedded_list) == features.shape[0])

    return features, embedded_list


def get_features2(model, loader, device):
    model.eval()

    features = None

    embedded_list = []
    #print('get_features, len:', len(embedding_list))
    s = time.time()

    total_iter = len(loader)
    for i, (imgs, labels, fs) in tqdm(enumerate(loader), total=total_iter) :
        imgs = imgs.to(device)
        outputs = model(imgs)

        labels = labels.data.cpu().numpy()
        cur_list = [{'query':f, 'label': label, 'dir': f.split('/')[-2]}
                for f, label in zip(fs, labels)]
        embedded_list += cur_list
        
        #embedded_list.append({'query':img_path, 'label': label})
        #print('i:', i, ', label:', label, ', path:', img_path)

        outputs = outputs.data.cpu().numpy()

        if features is None:
            features = outputs
        else:
            features = np.vstack((features, outputs))

    # print(len(embedded_list) , features.shape[0])
    assert(len(embedded_list) == features.shape[0])

    return features, embedded_list

def get_model(backbone):
    print('model:', backbone)
    if backbone == 'resnet18':
        model = resnet18(pretrained=False)
    elif backbone == 'resnet34':
        model = resnet34(pretrained=False)
    elif backbone == 'resnet50':
        model = resnet50(pretrained=False)
    return model


def get_distances(embedding_dic, test_features, test_embedded_list):
    # print(len(embedding_dic))
    ds = []
    ds2 = []
    for i in range(test_features.shape[0]):
        t_f = test_features[i]

        distance = []
        distance2 = {
            'query': os.path.splitext(test_embedded_list[i]['query'])[0].split('/')[-1],
            'dir': test_embedded_list[i]['dir']}
        for j, label in enumerate(embedding_dic):
            dic_name = embedding_dic[label]['dir']
            d = cosin_metric(t_f, embedding_dic[label]['features'])
            distance.append((label, d, dic_name))
            distance2[dic_name] = 1 - d
        # 거리가 1(max) 에 가까울 수록 유사 안경임
        ds2.append(distance2)
        distance = sorted(distance, key=lambda v: v[1], reverse=True)
        sorted_labels, sorted_distance, sorted_dir  = zip(*distance)
        ds.append({'query':test_embedded_list[i]['query'],
            'label': test_embedded_list[i]['label'],
            'distance': sorted_labels,
            'distance_': sorted_distance,
            'dir': sorted_dir})

    df = pd.DataFrame.from_records(ds)
    df.to_csv("distance.csv", index=False)
    df = pd.DataFrame.from_records(ds2)
    df.to_csv("distance2.csv", index=False)
    print('-- made distance.csv')
    return ds


def style_finder_test(model, opt):
    dataset = get_datasets(opt)
    loader = get_loaders(dataset, opt)
    device = torch.device('cuda')

    # embedding_image_list = get_list(opt.embedding_list, opt.embedding_root)
    # print('embedding_list count:', len(embedding_image_list))
    # test_image_list = get_list(opt.test_list, opt.test_root)
    # features, embedded_list = get_features1(model, embedding_image_list,
    #         batch_size=1, input_shape=opt.input_shape)
    # test_features, test_embedded_list = get_features1(model, test_image_list,
    #         batch_size=1, input_shape=opt.input_shape)

    s = time.time()

    features, embedded_list = get_features2(model, loader['embedding'], device)
    print('embedded_list count:', len(embedded_list))
    embedding_dic = get_feature_dict(embedded_list, features)
    print('embedding time is {}'.format(time.time() - s))
    print('embedding_dic count:', len(embedding_dic))

    s = time.time()
    test_features, test_embedded_list = get_features2(model, loader['test'], device)
    print('test image embedding time is {}'.format(time.time() - s))

    return get_distances(embedding_dic, test_features, test_embedded_list)


def calc_acc(result_image=False, distance=None, catalog_paths=None):
    def find(l, k):
        for i, v in enumerate(l):
            if int(v) == int(k):
                return i
        return None

    def load_image(img_path):
        image = PIL.Image.open(img_path)
        if image is None:
            print('none     ', img_path)
            return None
        image = image.resize((224,224))
        image = np.array(image)
        return image

    opt = StyleFinderConfig()
    embedding_list = get_list(opt.embedding_list, opt.embedding_root)
    embedding_list = {int(label):file for file, label in embedding_list}
    test_list = get_list(opt.test_list, opt.test_root)
    test_list = {query:label for query, label in test_list}

    if distance is None:
        fd = pd.read_csv('./distance.csv')
        ls = fd.to_dict('records')
        for i, v in enumerate(ls):
            if test_list[v['query']] != v['label']:
                print('error {}'.format(v['query']))
            distance = [d for d in v['distance'][1:-1].split(',')]
            ls[i]['distance'] = distance
    else:
        ls = distance

    top1_acc = 0
    top5_acc = 0
    top10_acc = 0
    ranks = []
    for i, v in enumerate(ls):
        if test_list[v['query']] != v['label']:
            print('error {}'.format(v['query']))

        distance = v['distance']
        rank = find(distance, v['label'])
        if rank is None:
            print('error, no rank', v['query'], ', ', v['label'], ', ', v)
        ranks.append(rank)
        v['rank'] = rank

        if rank == 0: top1_acc += 1
        if rank < 5 : top5_acc += 1
        if rank < 10 : top10_acc += 1


    #print('정답:', labels)
    top1 = top1_acc / len(ranks)
    top5 = top5_acc / len(ranks)
    top10 = top10_acc / len(ranks)
    print('top1 acc:', top1)
    print('top5 acc:', top5)
    print('top10 acc:', top10)
    print('성적:', sum(ranks)/len(ranks))
    print('성적:', ranks)

    if result_image:
        # 보기 좋게 정렬
        ls = sorted(ls, key=lambda v: v['query'].split('/')[-2], reverse=True)
        keys = [v['query'].split('/')[-2] for v in ls]

        fs = []
        for i, v in enumerate(ls):
            if test_list[v['query']] != v['label']:
                print('error {}'.format(v['query']))
            distance = v['distance']
            # print(v)

            files = [v['query']]
            labels = [v['label']] + distance[:10]
            for label in labels:
                dir_name = int(embedding_list[int(label)].split('/')[-2])
                if catalog_paths is not None and dir_name in catalog_paths:
                    files += [catalog_paths[dir_name]]
                else:
                    print('no in catalog', catalog_paths is None, label,
                            ', dir:', dir_name, ', ', embedding_list[int(label)])
                    files += [embedding_list[int(label)]]

            fs.append({'label': v['label'],
                       'rank': v['rank'],
                       'files':  files })

        save_root = os.path.expanduser('~/result')
        try:
            os.mkdir(save_root)
        except:
            pass

        for i, v in enumerate(fs):
            f = v['files']
            d = int(f[0].split('/')[-2])
            img = np.hstack((load_image(f_name) for f_name in f))
            img = PIL.Image.fromarray(img)
            img.save(os.path.join(save_root, '{:04d}_{:07d}_{:04d}.jpg'.format(v['rank'], d, i)))

        with open("result_path.txt", "w") as fd:
            for i, v in enumerate(fs):
                f = v['files']
                d = f[0].split('/')[-2]
                fd.write(d + ',' + ','.join(v['files']) + '\n')

    return top1, top5, top10


def load_checkpoint(model, ckpt_path):
    if ckpt_path and os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        
        model.load_state_dict(checkpoint)

        print("=> loaded checkpoint '{}' ".format(ckpt_path))
    else:
        print('failed to load_checkpoint:', ckpt_path)


def test(model, opt, result_image=False, catalog_paths=None):
    distance = style_finder_test(model, opt=opt)
    return calc_acc(result_image=result_image, distance=distance, catalog_paths=catalog_paths)


def catalog_path(root):
    imgs = glob.glob(root + "/*.jpg")
    imgs = {int(v.split('/')[-1].split('__')[-2]): os.path.join(root, v.split('/')[-1])
            for v in imgs}
    return imgs


def main():
    catalog_root = '/data/notebook/jongho/code/style_finder/catalog_caches'
    catalog_paths = catalog_path(catalog_root)
    # print(catalog_paths)

    calc_only = True
    if calc_only == False:
        opt = StyleFinderConfig()

        model = get_model(opt.backbone)
        model.to(torch.device('cuda'))
        model = DataParallel(model)
        load_checkpoint(model, opt.test_model_path)
        model.eval()

        test(model, opt, result_image=True, catalog_paths=catalog_paths)
    else:
        calc_acc(result_image=True, catalog_paths=catalog_paths)



if __name__ == "__main__":
    main()
