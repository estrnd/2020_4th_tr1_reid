import torch
import torch.nn as nn

from torchvision import models

import torch.nn as nn

from .resnetface import resnetface18, resnetface34, resnetface50


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, ckpt_path=None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnetface18":
        """ Resnet18
        """
        model_ft = resnetface18(pretrained=use_pretrained, num_classes=num_classes)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224

    elif model_name == "resnetface34":
        """ Resne34t
        """
        model_ft = resnetface34(pretrained=use_pretrained, num_classes=num_classes)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224

    elif model_name == "resnetface50":
        """ Resne50t
        """
        model_ft = resnetface50(pretrained=use_pretrained, num_classes=num_classes)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name.startswith("squeezenet1.1"):
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name.startswith("squeezenet"):
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        # if ckpt_path:
        #     ckpt = torch.load(ckpt_path)

        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in ckpt.items():
        #         name = k[7:] # remove 'module.' of dataparallel
        #         new_state_dict[name] = v
        #     model_ft.load_state_dict(new_state_dict)

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



def resnet18(pretrained=False, ckpt_path=None, num_classes=512, **kwargs):
    model, _ = initialize_model("resnetface18",
                                num_classes=512,
                                feature_extract=False,
                                use_pretrained=pretrained,
                                ckpt_path=ckpt_path)
    return model


def resnet34(pretrained=True, ckpt_path=None, **kwargs):
    model, _ = initialize_model("resnetface34",
                                num_classes=512,
                                feature_extract=False,
                                use_pretrained=pretrained,
                                ckpt_path=ckpt_path)
    return model


def resnet50(pretrained=True, ckpt_path=None, **kwargs):
    model, _ = initialize_model("resnetface50",
                                num_classes=512,
                                feature_extract=False,
                                use_pretrained=pretrained,
                                ckpt_path=ckpt_path)
    return model


def main():
    model = resnet18()
    print(model)


if __name__ == '__main__':
    main()
