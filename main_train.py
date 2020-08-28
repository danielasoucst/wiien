# based on the code from https://github.com/cszn/DnCNN
# -*- coding: utf-8 -*-
import argparse
import re
import os, glob, datetime, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import LightingDataset
import torch.backends.cudnn as cudnn
import random
from config import *
from light_model import LightNet
from torch.optim.lr_scheduler import StepLR
from utils import *
from torchvision import transforms, utils
import scipy.io as sio

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# model_name = 'WIIEN_sgd_seed' + str(SEED)
model_name = 'model_100.pth'
# Params
parser = argparse.ArgumentParser(description='PyTorch Light Net')
parser.add_argument('--model', default=model_name, type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('--train_data', default=TRAIN_DATASET, type=str, help='path of train data')
parser.add_argument('--val_data', default=VALIDATION_DATASET, type=str, help='path of validation data')
parser.add_argument('--epoch', default=100, type=int, help='number of train epoches')
parser.add_argument('--lr', default=LEARNING_RATE, type=float, help='initial learning rate for Adam')
parser.add_argument('--seed', default=SEED, type=int, help='seed')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch

"""Determinismo"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True


def worker_init(worker_id):
    random.seed(args.seed)


save_dir = os.path.join(MODEL_PATH, args.model)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# class sum_squared_error(_Loss):  # PyTorch 0.4.1
#     """
#     Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
#     The backward is defined as: input-target
#     """
#     def __init__(self, size_average=None, reduce=None, reduction='sum'):
#         super(sum_squared_error, self).__init__(size_average, reduce, reduction)

#     def forward(self, input, target):
#         # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
#         return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def validation(epoch, val_DLoader):  # val_DDataset,

    val_loss = 0
    start_time = time.time()

    for n_count, batch_yx in enumerate(val_DLoader):
        optimizer.zero_grad()
        if cuda:
            batch_x, batch_y, gd = batch_yx[0].cuda(), batch_yx[1].cuda(), batch_yx[2]

        output, illu_estim = model(batch_y)
        ssim_value = torch.mean(ssim(batch_x, output, data_range=1, size_average=False))
        loss = 1 - ssim_value

        val_loss += loss.item()

    elapsed_time = time.time() - start_time

    print('epoch val = %4d , loss = %4.10f , time = %4.2f s' % (
    epoch + 1, val_loss / (n_count * batch_size), elapsed_time))

    f = open(save_dir + '/validation_result.txt', "a+")
    if f is not None:
        f.write('epoch = %4d , loss = %4.10f , lr = %2.4f, time = %4.2fs \n' % (
        epoch + 1, val_loss / (n_count * batch_size), LEARNING_RATE, elapsed_time))
    f.close()


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = LightNet()

    transform_train = transforms.Compose([
        # transforms.RandomAffine(0, scale=(250,1000)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.43275916, 0.43697679, 0.36495983), (0.21435698, 0.2006122, 0.19247726))
    ])

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    # criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = sum_squared_error()
    # criterion =  pytorch_ssim.SSIM(window_size = 8)

    ssim_module = SSIM(win_size=11, win_sigma=1.5, data_range=1, K=(0.01, 0.8), size_average=True, channel=3)

    # criterion = nn.L1Loss()

    if cuda:
        model = model.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        # criterion = criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = MultiStepLR(optimizer, milestones=[36, 60, 90], gamma=0.2)  # learning rates

    # DADOS PARA VALIDACAO DO TREINAMENTO
    # val_xs, val_illums = dg.datagenerator(data_dir=args.val_data, batch_size=args.batch_size, is_validation=True)
    # val_xs = val_xs.astype(np.float32)/255.0 #float = [0,1]
    # val_xs = torch.from_numpy(np.reshape(val_xs,(val_xs.shape[0], val_xs.shape[3], val_xs.shape[1], val_xs.shape[2])))
    # print('val_xs',val_xs.shape)
    # val_DDataset = DenoisingDataset(val_xs, val_illums)
    # val_DLoader = DataLoader(dataset=val_DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True,
    #                          worker_init_fn=worker_init)

    for epoch in range(initial_epoch, n_epoch):
        print('Treinando epoca %d' % (epoch))
        # print('Treinando epoca %d com fator %f'%(epoch, fatores[epoch]))
        # scheduler.step(epoch)

        xs, illums = dg.datagenerator(data_dir=args.train_data, batch_size=args.batch_size)

        xs = xs.astype(np.float32) / 255.0  # float = [0,1]

        xs = torch.from_numpy(np.reshape(xs, (
        xs.shape[0], xs.shape[3], xs.shape[1], xs.shape[2])))  # tensor of the clean patches, NXCXHXW

        # xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        # xs = xs.transpose((0, 3, 1, 2))

        DDataset = LightingDataset(xs, illums)

        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True,
                             worker_init_fn=worker_init)

        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
            optimizer.zero_grad()
            if cuda:
                batch_x, batch_y, gd = batch_yx[0].cuda(), batch_yx[1].cuda(), batch_yx[2]
                # batch_x, batch_y = batch_yx['clean_image'].cuda(), batch_yx['dark_image'].cuda()
            # print('batch shape ', batch_x.shape) #(batch_size, 3, 40,40)
            output, illu_estim = model(batch_y)


            # loss = 1 - ssim_loss(batch_x, output) # sum_squared
            ssim_value = torch.mean(ssim(batch_x, output, data_range=1, size_average=False))
            loss = 1 - ssim_value
            # print('loss ssim: ', ssim_value)

            # print('loss mean : ', loss))

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 100 == 0:
                print('%4d %4d / %4d loss = %2.10f' % (
                epoch + 1, n_count, len(xs) // batch_size, loss.item() / batch_size))

                # print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time

        print('epcoh = %4d , loss = %4.10f , time = %4.2f s' % (
        epoch + 1, epoch_loss / (n_count * batch_size), elapsed_time))

        f = open(save_dir + '/train_result.txt', "a+")
        if f is not None:
            f.write('epoch = %4d , loss = %4.10f , lr = %2.4f, time = %4.2fs \n' % (
            epoch + 1, epoch_loss / (n_count * batch_size), LEARNING_RATE, elapsed_time))
        f.close()

        # validation(epoch, val_DLoader)

        # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # print('salvou model_%03d.pth' % (epoch+1))

        if (epoch + 1) % 5 == 0:
            print('Saving CHECKPOINT...')
            torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))

        del xs
        del DDataset
        del DLoader
        torch.cuda.empty_cache()

