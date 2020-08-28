# -*- coding: utf-8 -*-
import argparse
import os, time, datetime
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from data_generator import LightingDataset
import random
from config import *
from sklearn.metrics import mean_squared_error
import utils
# import statistics
from scipy import stats

"""set_names : 'synthetic_kodak' """

model_name = 'model_100.pth'
random.seed(SEED)  # for reproducibility


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default=DATASET_TEST_DIR, type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['synthetic_kodak'], help='directory of test dataset')
    parser.add_argument('--model_dir', default=MODEL_PATH, help='directory of the model')
    parser.add_argument('--model_name', default='model_100.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default=RESULT_DIR, type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the result image, 1 or 0')
    return parser.parse_args()


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        # imsave(path, np.clip(result, 0, 1))
        e_img = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(e_img, cv2.COLOR_BGR2RGB))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':

    args = parse_args()

    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        print('Model not found!')
        quit()
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model = torch.load(os.path.join(args.model_dir, args.model_name))
        print('load trained model')

    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(os.path.join(args.result_dir, model_name)):
        os.mkdir(os.path.join(args.result_dir, model_name))

    for set_cur in args.set_names:

        print('Testing %s dataset' % set_cur)
        if not os.path.exists(os.path.join(args.result_dir, model_name, set_cur)):
            os.mkdir(os.path.join(args.result_dir, model_name, set_cur))

        psnrs = []
        ssims = []

        file_list = os.listdir(os.path.join(args.set_dir, set_cur))
        for k in range(1, len(file_list) + 1):
            im = str(k) + '.png'
            if im.endswith(".JPG") or im.endswith(".bmp") or im.endswith(".png"):

                x = np.array(imread(os.path.join(args.set_dir, 'kodak', im)), dtype=np.float32) / 255.0
                y = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0

                x = cv2.resize(x, (600, 600))
                y = cv2.resize(y, (600, 600))

                patches_x = utils.blockshaped(x, 15, 15)
                patches_y = utils.blockshaped(y, 15, 15)

                patches_restaured = []
                illums = []
                for i in range(len(patches_y)):
                    patch = patches_y[i]
                    y_ = torch.from_numpy(np.reshape(patch, (1, patch.shape[2], patch.shape[0], patch.shape[1])))
                    torch.cuda.synchronize()
                    start_time = time.time()
                    y_ = y_.cuda()
                    x_, illu = model(y_)  # inference

                    x_ = x_.view(patch.shape[0], patch.shape[1], 3)
                    x_ = x_.cpu()
                    x_ = x_.detach().numpy().astype(np.float32)
                    patches_restaured.append(x_)
                    illums.append(illu.item())

                # print('patches_restaured', np.array(patches_restaured).shape)
                y_restaured, illu_map = utils.recompose_image(np.array(patches_restaured), illums)
                # Global illumination
                iilu_g = np.mean(np.array(illums))
                y_restaured = y / iilu_g

                # apply metrics
                psnr_x_ = compare_psnr(x, y_restaured)
                ssim_x_ = compare_ssim(x, y_restaured, multichannel=True)

                print("image: %s PSNR = %2.2f SSIM = %2.4f" % (im, psnr_x_, ssim_x_))

                if args.save_result:
                    name, ext = os.path.splitext(im)
                    save_result(y_restaured, path=os.path.join(args.result_dir, model_name, set_cur, im))

                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)

        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        if args.save_result:
            save_result(np.hstack((psnrs, ssims)),
                        path=os.path.join(args.result_dir, model_name, set_cur, 'results.txt'))
        print('PSNR = %2.2f SSIM = %2.4f ' % (psnr_avg, ssim_avg))

