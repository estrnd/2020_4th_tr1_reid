import imgaug as ia
from imgaug import augmenters as iaa
import torchvision.transforms as T
import numpy as np

def get_transform(mode='train'):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    normalize = T.Normalize(mean=MEAN, std=STD)
    hw = iaa.Scale({'height': 224, 'width':"keep-aspect-ratio"}, ia.ALL)
    wh = iaa.Scale({'width': 224, 'height':"keep-aspect-ratio"}, ia.ALL)

    def fit_size(x):
        x = np.array(x)
        if x.shape[0] > x.shape[1]:
            return hw.augment_image(x)
        else:
            return wh.augment_image(x)

    if mode=='train':
        train_iaa = iaa.Sequential([

            iaa.OneOf([
                iaa.PerspectiveTransform(scale=0.07),
                iaa.Affine(rotate=(-20, 20)),
                # iaa.CropAndPad(percent=(-0.25,0.25)),
            ]),
            iaa.Sometimes(0.6,
                          iaa.OneOf([
                              iaa.contrast.GammaContrast(gamma=(0.5, 1.75)),
                              iaa.LogContrast(gain=(0.6, 0.8), per_channel=True),
                              iaa.SigmoidContrast(gain=(3, 5), cutoff=(0.25, 0.5), per_channel=True),
                          ])
                          ),

            iaa.Fliplr(p=0.5),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.0))),
            iaa.Sometimes(0.2, iaa.OneOf([
                iaa.AdditiveLaplaceNoise(scale=(255 * 0.02, 255 * 0.06)),
                iaa.AdditiveGaussianNoise(scale=(255 * 0.03, 255 * 0.06)),
                iaa.AdditivePoissonNoise((0, 3))])),
            iaa.PadToFixedSize(224, 224, position='center'),
        ])

        train_transform = T.Compose([
            fit_size,
            train_iaa.augment_image,
            T.ToTensor(),
            normalize
        ])
        return train_transform

    else:
        valid_iaa = iaa.Sequential([
            iaa.PadToFixedSize(224, 224, position='center'),
        ])

        valid_transform = T.Compose([
            fit_size,
            valid_iaa.augment_image,
            T.ToTensor(),
            normalize
        ])
        return valid_transform
