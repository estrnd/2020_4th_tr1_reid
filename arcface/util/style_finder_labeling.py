import os
import glob
import pandas as pd
import random


def get_test_image_list(file_list, dir_label):
    l = []
    for f in file_list:
        d, _ = f.split('/')
        if d in dir_label:
            l.append((f, dir_label[d]))
        else:
            print('unknown directory:', d)

    return l 


def get_embedding_image_list(root, count_per_id):
    root = os.path.expanduser(root)
    imgs = glob.glob(root + "/**/*.jpg")
    imgs = [os.path.join(v.split('/')[-2], v.split('/')[-1]) for v in imgs]

    dirs = [v.split('/')[-2] for v in imgs]
    dir_label_list = {d: label for label, d in enumerate(set(dirs))}
    # print('-------------- ', dir_label_list['180552'])
    imgs = [(f, dir_label_list[f.split('/')[-2]]) for f in imgs]

    random.shuffle(imgs)
    embedding_list = {}
    for f, label in imgs:
        label = int(label)
        if label not in embedding_list:
            embedding_list[label] = []
        if len(embedding_list[label]) < count_per_id:
            embedding_list[label].append(f)

    sorted_key = sorted(embedding_list)
    embedding_list = [(embedding_list[k][i], k) 
            for k in sorted_key for i in range(len(embedding_list[k])) ]
    return embedding_list, dir_label_list


def get_files(root):
    root = os.path.expanduser(root)
    imgs = glob.glob(root + "/**/*.jpg")
    imgs = [os.path.join(v.split('/')[-2], v.split('/')[-1]) for v in imgs]
    return imgs


def save_embedding_list(root, count_per_id=3):
    l, dir_label_list = get_embedding_image_list(root, count_per_id=count_per_id)
    df = pd.DataFrame.from_records(l)
    df.to_csv("embedding_list_testset_1st_emb_v2_"+str(count_per_id)+".txt",
            index=False, header=['path','label'])
    return dir_label_list


def save_test_list(test_root, dir_label_list):
    l = get_test_image_list(get_files(test_root),
                            dir_label_list)
    l = sorted(l, key=lambda v: v[0].split('/')[-2])
    df = pd.DataFrame.from_records(l)
    df.to_csv("test_testset_1st_emb_v2.txt", index=False, header=['path','label'])


def main():
    # embedding_images_root = '~/dataset/stylefinder/style_finder_data/stylefinder_v1'
    # test_root = '~/dataset/stylefinder/rf_test_crop'
    embedding_images_root = '~/dataset/stylefinder/style_finder_data/v2_crop'
    # test_root = '~/dataset/stylefinder/rf_test_crop'
    # embedding_images_root = '~/dataset/stylefinder/testset_1st_crop/reg'
    test_root = '~/dataset/stylefinder/testset_1st_crop/query'

    dir_label_list = save_embedding_list(embedding_images_root, count_per_id=3)

    save_test_list(test_root, dir_label_list) 


if __name__ == '__main__':
    main()



