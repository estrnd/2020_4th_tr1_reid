import os
import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from .transforms import get_transform

def rect_pad(pil_image):
    import numpy as np
    from PIL import Image
    
    im = np.array(pil_image) 
    h, w = im.shape[:2]
    size = max(h, w)
    pad_h, pad_w = size-h, size -w
    im = np.pad(im, 
           ((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w - pad_w//2), (0,0)), 
           mode='constant', constant_values=0)
    return Image.fromarray(im)

class DataSet(data.Dataset):

    def __init__(self, root, data_list_file, phase='train',
            input_shape=(3,224,224), model_type='face',
            transforms=None):
        # self.MEAN = [0.5, 0.5, 0.5]
        # self.STD  = [0.5, 0.5, 0.5]
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD  = [0.229, 0.224, 0.225]


        self.phase = phase
        self.input_shape = input_shape
        self.root = root

        # with open(os.path.join(data_list_file), 'r') as fd:
        #     imgs = fd.readlines()
        df = pd.read_csv(data_list_file)
        imgs = df.to_dict('records')
        # if phase == 'train' or phase == 'val':
        #     imgs = imgs[:1000]

        # imgs = [(os.path.join(root, (v['path'])), v['label']) for v in imgs]

        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=self.MEAN, std=self.STD)
        image_scale = int(input_shape[1] * 1.125)

        transforms_origin = T.Compose([
                T.RandomResizedCrop(self.input_shape[1],
                                    scale= (0.85, 0.99)),
                T.RandomHorizontalFlip(),
                # T.ColorJitter(),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                    hue=0.01),
                T.ToTensor(),
                normalize
                ])
        if self.phase == 'train':
            if transforms is not None:
                self.transforms = transforms(self.phase)
        else:
            self.transforms = T.Compose([
                T.Resize(image_scale),
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize,
            ]) 


    def __getitem__(self, index): 
        try:
            return self._getitem(index)
        except Exception as e:
            print('occur exception !!!!')
            print('Exception:', e)
            return self.__getitem__(index+1)
    
    def _getitem(self, index):
        sample = self.imgs[index]
        if self.root is None:
            img_path = sample['NAME']
        else:
            img_path = os.path.join(self.root, sample['NAME'])

        data = Image.open(img_path)
        data = rect_pad(data)

        data = self.transforms(data)

        # print(img_path, label, self.phase)
        label = sample['KLS_IDX']
        return data.float(), int(label), img_path

    def __len__(self):
        return len(self.imgs)


def get_datasets(opt):
    dataset = {
        'train': DataSet(opt.train_root, opt.train_list, phase='train',
                         input_shape=opt.input_shape, model_type=opt.type,
                         transforms=get_transform),
        'val'  : DataSet(opt.train_root, opt.val_list, phase='val',
                         input_shape=opt.input_shape, model_type=opt.type,
                         transforms=get_transform),
        'test'  : DataSet(opt.test_root, opt.test_list, phase='test',
                         input_shape=opt.input_shape, model_type=opt.type,
                         transforms=get_transform),
        'embedding'  : DataSet(opt.embedding_root, opt.embedding_list, phase='embedding',
                         input_shape=opt.input_shape, model_type=opt.type,
                         transforms=get_transform),
    }
    return dataset


def get_loaders(dataset, opt):
    batch_size = {'train':      opt.train_batch_size,
                  'val':        opt.val_batch_size,
                  'test':       opt.test_batch_size,
                  'embedding':  opt.test_batch_size}
    shuffle = {'train':True,
               'val':False,
               'test':False,
               'embedding':False}

    loader = {
        x: data.DataLoader(dataset[x],
                           batch_size=batch_size[x],
                           shuffle=shuffle[x],
                           num_workers=opt.num_workers)
        for x in ['train', 'val', 'test', 'embedding']}
    return loader


def main():
    import matplotlib.pyplot as plt
    from config_style_finder import StyleFinderConfig
    from config import Config

    #opt = Config()
    opt = StyleFinderConfig()

    root = {'train':opt.train_root,
            'val':opt.train_root,
            'test':opt.test_root,
            'embedding':opt.embedding_root}
    file_list = {'train': opt.train_list,
                'val': opt.val_list,
                'test': opt.test_list,
                'embedding': opt.embedding_list}
    root = {k:os.path.expanduser(root[k]) for k in root}
    file_list = {k:os.path.expanduser(file_list[k]) for k in file_list}

    #for phase in ['train', 'val', 'test']:
    for phase in ['train']:
        print('root ', root, ', phase:', phase)

        dataset = DataSet(root=root[phase],
                          data_list_file=file_list[phase],
                          phase=phase,
                          input_shape=opt.input_shape,
                          model_type=opt.type,
                          transforms=get_transform)
        # MEAN = [0.5, 0.5, 0.5]
        # STD  = [0.5, 0.5, 0.5]
        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=100)
        for i, v in enumerate(trainloader):
            if phase == 'test':
                data, label, img_path = v[0], v[1], v[2]
            else:
                data, label = v[0], v[1]

            img = torchvision.utils.make_grid(data, nrow=10).numpy()
            # chw -> hwc
            img = np.transpose(img, (1,2,0))
            img = np.array(STD) * img + np.array(MEAN)
            img = np.clip(img, 0, 1)
            img *= np.array([255,255,255])
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.show()
            break

if __name__ == '__main__':
    main()


