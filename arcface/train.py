from __future__ import print_function
import os

from data import get_datasets, get_loaders
from models import *
import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision
from utils import Visualizer
import numpy as np
import random
import time
from config import Config
from config_reid import ReIDConfig as  train_config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from test import *
import test_style_finder as test_sf
from tqdm import tqdm

from tensorboardX import SummaryWriter


def save_model(model, save_path, name, iter_cnt):
    try:
        print('mkdir :', './' + save_path)
        os.mkdir('./' + save_path)
    except:
        pass
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def mask_to_index(mask):
    import itertools
    return [v[1] for v in zip(mask, itertools.count()) if v[0]==True]


def train(model, optimizer, loader, device, opt,
          metric_fc, 
          criterion, summary, phase, epo, scheduler):
    model.train()

    start = time.time()
    total_iter = len(loader)
    for i, (imgs, labels, _) in tqdm(enumerate(loader), total=len(loader)) :

        imgs     = imgs.to(device)
        labels   = labels.to(device).long()
        # glass_models   = glass_models.to(device).long()

        features = model(imgs)
        outputs  = metric_fc(features, labels)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iters = epo * total_iter + i + 1

        if iters % opt.print_freq == 0 or iters % (len(loader) - 1) == 0:
            outputs = outputs.data.cpu().numpy()
            # outputs_ori = outputs.copy()
            outputs = np.argmax(outputs, axis=1)
            labels = labels.data.cpu().numpy()
            acc = np.mean((outputs == labels).astype(float))
            # print('labels:', labels)
            # print('output:', outputs)
            # print('none?:', np.any(outputs_ori == None))
            # print('compare:', outputs == labels)
            # temp = outputs_ori[0,:]
            # print('************ outputs:', outputs_ori[0,:5],
            #         '\n, output[0,label]:', outputs_ori[0,labels.data[0]],
            #         '\n, output[0,out]:', outputs_ori[0,outputs[0]],
            #         '\n, mean output:', np.mean(outputs_ori[0,:]),
            #         '\n, over 0.9 count:', len(temp[temp > 0.9]))
            speed = opt.print_freq / (time.time() - start)
            time_str = time.asctime(time.localtime(time.time()))
            print('{} {} epoch {} iter {} {} iters/s loss {} acc {} '.format(
                  time_str, phase, epo, iters, speed, loss.item(), acc))

            if summary is not None:
                summary.add_scalar(phase+'/loss', loss.item(), iters)
                summary.add_scalar(phase+'/acc', acc, iters)
                summary.add_scalar(phase+'/lr', scheduler.get_lr()[0], iters)

            start = time.time()


def val(model, loader, device, opt,
        metric_fc, 
        summary, phase, epo, dataset):
    model.eval()
    acc_count= 0
    acc_count_other = [0,0,0]
    total_num = len(dataset)
    start = time.time()
    total_iter = len(loader)
    for i, (imgs, labels, _) in tqdm(enumerate(loader), total=total_iter) :

        imgs     = imgs.to(device)
        labels   = labels.to(device).long()

        features = model(imgs)
        outputs  = metric_fc(features, labels)
        
        outputs  = np.argmax(outputs.data.cpu().numpy(), axis=1)
        
        labels = labels.data.cpu().numpy()
        acc_count += np.sum((outputs == labels).astype(float))

        iters = epo * total_iter + i + 1

        if iters % opt.print_freq == 0:
            speed = opt.print_freq / (time.time() - start)
            acc = float(acc_count) / (iters * opt.train_batch_size)
            time_str = time.asctime(time.localtime(time.time()))
            print('{} {} epch {} iter {} {} iters/s acc {}, acc_count {}, total {}'.format(
                  time_str, phase, epo, i, speed, acc, acc_count, total_num))
            start = time.time()

    acc = float(acc_count) / float(total_num)
    acc_others = [float(acc) / float(total_num) for acc in acc_count_other]
    print('{} epch {} acc {}, acc_count {}, total {}'.format(
           phase, epo, acc, acc_count, total_num))
    return acc, acc_others


def get_model(backbone):
    if backbone == 'resnet18':
        model = resnet18(pretrained=True)
        print('model:', backbone)
    elif backbone == 'resnet34':
        model = resnet34(pretrained=True)
    elif backbone == 'resnet50':
        model = resnet50(pretrained=True)
    #elif backbone == 'bit':
        #model = big_transfer(pretrained=True)

    return model


def load_checkpoint(model, ckpt_path):
    if ckpt_path and os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        
        model.load_state_dict(checkpoint)

        print("=> loaded checkpoint '{}' ".format(ckpt_path))
    else:
        print('failed to load_checkpoint:', ckpt_path)


def get_criterion(loss):
    if loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion



def get_optimizer(opt, model, metric_fc):
    if opt.optimizer == 'sgd':
        # optimizer = torch.optim.SGD([{'params': model.parameters()},
        #                              {'params': metric_fc.parameters()}],
        #                             lr=opt.lr,
        #                             weight_decay=opt.weight_decay)
        params = add_weight_decay(model, opt.weight_decay)
        params += add_weight_decay(metric_fc, opt.weight_decay)
        optimizer = torch.optim.SGD(params, lr=opt.lr)
    else:
        # optimizer = torch.optim.Adam([{'params': model.parameters()},
        #                               {'params': metric_fc.parameters()}],
        #                              lr=opt.lr,
        #                              weight_decay=opt.weight_decay)
        params = add_weight_decay(model, opt.weight_decay)
        params += add_weight_decay(metric_fc, opt.weight_decay)
        optimizer = torch.optim.Adam(params, lr=opt.lr)
    #print('model parameter len:', len(model.parameters()))
    # for param in metric_fc.parameters():
    #     print('metric_fc:', type(param.data), param.size())
    return optimizer


def main():
    #opt = Config()
    opt = train_config()
    device = torch.device("cuda")


    dataset = get_datasets(opt)
    loader = get_loaders(dataset, opt)
    
    metric_fc = ArcMarginProduct(512, opt.num_classes, s=64, m=0.5,
                                 easy_margin=opt.easy_margin)
    
    criterion = get_criterion(opt.loss)
    model = get_model(opt.backbone)
    freez_model(opt.backbone, model, opt.freez_layers)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    optimizer = get_optimizer(opt, model, 
                              metric_fc=metric_fc)
    opt.milestones = [epo - opt.start_epoch for epo in opt.milestones]
    print('milestones:', opt.milestones)
    scheduler = MultiStepLR(optimizer, opt.milestones, gamma=0.1)
                            #last_epoch=opt.start_epoch)

    if opt.load_model_path is not None:
        load_checkpoint(metric_fc,
                os.path.join(opt.checkpoints_path, 'metric_fc_' + opt.load_model_path))
        
        # load_checkpoint(metric_model,
        #         os.path.join(opt.checkpoints_path, 'metric_model_' + opt.load_model_path))
        # load_checkpoint(optimizer,
        #         os.path.join(opt.checkpoints_path, 'optimizer_' + opt.load_model_path))

    identity_list = None
    img_paths = None
    if opt.lfw_test_list is not None:
        identity_list = get_lfw_list(opt.lfw_test_list)
        img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]
    summary = SummaryWriter()
    #summary = None
    start = time.time()
    acc = 0
    epo = 0
    if epo % opt.save_interval == 0 or epo == opt.max_epoch:
        save_model(model, opt.checkpoints_path, opt.backbone, epo)

    for epo in range(opt.start_epoch, opt.max_epoch):
        for phase in ['train', 'val']:
            # print('############ ', phase)
            if phase == 'train':
                print('epo {} lr {}'.format(epo, scheduler.get_last_lr()[0]))
                train(model=model, optimizer=optimizer, scheduler=scheduler,
                      loader=loader[phase],
                      device=device, opt=opt, 
                      metric_fc=metric_fc,
                      criterion=criterion, summary=summary, phase=phase,
                      epo=epo)
            elif phase == 'val':
                acc, acc_others = val(model=model, loader=loader[phase], device=device,
                          opt=opt, 
                          metric_fc=metric_fc,
                          summary=summary,
                          phase=phase, epo=epo, dataset=dataset[phase])
                # print('------------------------ ', (summary is not None),
                # ', acc:', acc, ', phase:', phase, ', epo:', epo)
                if summary is not None:
                    summary.add_scalar(phase+'/acc', acc, epo)

            elif opt.type == 'style_finder' and phase == 'test':
                model.eval()
                top1_acc, top5_acc, top10_acc = test_sf.test(model, opt, result_image=False)
                print('test - top1_acc:', top1_acc, ', epo:', epo)
                if summary is not None:
                    print('------------------------ ', (summary is not None),
                          ', top1_acc:', top1_acc, ', phase:', phase, ', epo:', epo)
                    summary.add_scalar(phase+'/top1_acc', top1_acc, epo)
                    summary.add_scalar(phase+'/top5_acc', top5_acc, epo)
                    summary.add_scalar(phase+'/top10_acc', top10_acc, epo)
        scheduler.step()
        
        if epo % opt.save_interval == 0 or epo == opt.max_epoch:
            save_model(model,        opt.checkpoints_path, opt.backbone, epo)
            save_model(metric_fc,    opt.checkpoints_path, 'metric_fc_' + opt.backbone, epo)
            save_model(optimizer,    opt.checkpoints_path, 'optimizer_' + opt.backbone, epo)
            save_model(scheduler,    opt.checkpoints_path, 'scheduler_' + opt.backbone, epo)


if __name__ == '__main__':
    main()
