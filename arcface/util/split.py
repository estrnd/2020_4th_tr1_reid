import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def lfw():
    root = '~/dataset/lfw_crop'
    root = os.path.expanduser(root)
    imgs = glob.glob(root + "/**/*.jpg")
    imgs = [(os.path.join(v.split('/')[-2], v.split('/')[-1]),) for v in imgs]
    print(imgs[:5])
    
    columns = ['path']
    df = pd.DataFrame.from_records(imgs)
    df.to_csv("./lfw_total.txt", index=False, header=False, sep=',')


def to_label(ls):
    ls = set(ls)
    print('unique:', len(ls))
    labels = {k:i for i, k in enumerate(ls)}
    return labels


def read_meta(meta_path):
    def convert_frame(v):
        v = str(v)
        # 공백제거
        v = v.replace(' ', '').replace(',', '').replace('/', '')
        if v.find('투명') >= 0: return 'trans'
        if v.find('무테') >= 0 or v.find('반무테') >= 0: return 'frameless'
        if v.find('하금테') >= 0: return  'bottom'
        if v.find('뿔테') >= 0 : return 'bbul'
        if v.find('일반메탈') == 0 or v.find('티타늄') == 0 or v.find('메탈') == 0: return 'metal'
        if v == 'nan'   : return 'none'
        return 'none'

    def convert_lens(v):
        v = str(v)
        # 공백제거
        v = v.replace(' ', '').replace(',', '').replace('/', '')
        if v.find('블랙') >= 0   : return 'lens_black'
        if v.find('브라운') >= 0 : return 'lens_brown'
        if v.find('틴트') >= 0   : return 'lens_tint'
        if v.find('미러') >= 0   : return 'lens_mirror'
        if v.find('그외') >= 0 or v.find('기타') >= 0  : return 'etc'
        if len(v) == 0 or v == 'nan'   : return 'lens_trans'
        return 'none'

    def convert_model_number(v):
        v = str(v)
        return v.split('-')[0]

    df = pd.read_csv(meta_path, 
                     usecols=['product_index', 'prop_frame', 'prop_lenscolor', 'model_number'])
    ls = df.to_dict('records')
    # print(ls[:10])

    results = {}
    # results_list = []
    for v in ls:
        results[v['product_index']] = {
               'model': convert_model_number(v['model_number']),
               'lens':  convert_lens(v['prop_lenscolor']),
               'frame':  convert_frame(v['prop_frame']) }

    model_labels = to_label([results[k]['model'] for k in results] + ['none'])
    frame_labels = to_label([results[k]['frame'] for k in results] + ['none'])
    lens_labels  = to_label([results[k]['lens'] for k in results] + ['none'])
    print('model_labels count: ', len(model_labels))
    print('frame_labels count: ', len(frame_labels))
    print('lens_labels count: ', len(lens_labels))
    results = { k: {
        'model_origin': results[k]['model'],
        'lens_origin': results[k]['lens'],
        'frame_origin': results[k]['frame'],
        'model':model_labels[results[k]['model']],
        'lens': lens_labels[results[k]['lens']],
        'frame': frame_labels[results[k]['frame']]}
        for k in results}

    # df = pd.DataFrame.from_records(results_list)
    # df.to_csv('meta.csv', index=False)
    return results, model_labels, frame_labels, lens_labels
    

def empty_meta(model_labels, frame_labels, lens_labels):
    return { 'model_origin': 'none',
        'lens_origin': 'none',
        'frame_origin': 'none',
        'model':model_labels['none'],
        'lens':lens_labels['none'],
        'frame': frame_labels['none']}


def split(roots, name, meta, model_labels, frame_labels, lens_labels, except_dir=[]):
    imgs = []
    for root in roots:
        imgs += glob.glob(root + "/**/*.jpg")
        print(len(imgs))
    #imgs = [os.path.join(v.split('/')[-2], v.split('/')[-1]) for v in imgs]

    folders = [int(v.split('/')[-2]) for v in imgs]
    labels = to_label(folders)
    print('labels count: ', len(labels))

    ls = []
    for img, folder in zip(imgs, folders):
        if folder in except_dir: continue
        if folder not in meta: continue
        #     ls.append(dict(
        #         {'path':img, 'label':labels[folder]},
        #         **empty_meta(model_labels, frame_labels, lens_labels)))
        ls.append(dict({'path':img, 'label':labels[folder]}, **meta[folder]))
    #print(imgs[:10])
    print('no meta samples : ', len(imgs) - len(ls))
    imgs = ls

    df = pd.DataFrame.from_records(imgs)
    df.to_csv("./{}_total.txt".format(name),
                 index=False, sep=',')

    train, test = train_test_split(df, test_size=0.1, shuffle=True,
                                   random_state=100)
    train.to_csv("./{}_train.txt".format(name),
                 index=False, sep=',')
    test.to_csv("./{}_val.txt".format(name),
                 index=False, sep=',')


def get_folders(root):
    root = os.path.expanduser(root)
    return os.listdir(root)


if __name__ == '__main__':
    # root = '~/dataset/CASIA-maxpy-clean_crop2'
    # name = 'casia'
    # root = '~/dataset/stylefinder/style_finder_data/stylefinder_v1'
    # name = 'style_finder'
    roots = ['/home/snowyunee/dataset/stylefinder/v2',
             '/home/snowyunee/dataset/stylefinder/v3_crop']
    name = 'vf'
    meta = '/home/snowyunee/dataset/stylefinder/meta.csv'
    meta, model_labels, frame_labels, lens_labels = read_meta(meta)

    # test_root = '~/dataset/stylefinder/rf_test_crop'
    # test_labels = get_folders(test_root)
    try:
        for v in ['total', 'train', 'val']:
            os.remove('./{}_{}.txt'.format(name, v))
    except: pass
    split(roots=roots, name=name, meta=meta,
            model_labels=model_labels,
            frame_labels=frame_labels,
            lens_labels=lens_labels)

