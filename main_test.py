# -*- coding: utf-8 -*-
# based on the code from https://github.com/cszn/DnCNN

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
import loe_metric as loe

random.seed(0)  # for reproducibility

model_name = 'model_100.pth'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default=DATASET_TEST_DIR + '/bases_reais', type=str,
                        help='directory of test dataset')
    parser.add_argument('--set_names', default=['LIME', 'NPE', 'DICM', 'MEF', 'VV'], help='directory of test dataset')
    parser.add_argument('--model_dir', default=MODEL_PATH, help='directory of the model')
    parser.add_argument('--model_name', default='model_100.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default=RESULT_DIR, type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the final image, 1 or 0')
    return parser.parse_args()


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        f = open(path, "a+")
        # if f is not None:
        f.close()
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
        print('Dataset: ', set_cur)
        if not os.path.exists(os.path.join(args.result_dir, model_name, set_cur)):
            os.mkdir(os.path.join(args.result_dir, model_name, set_cur))

        loes = []
        file_list = os.listdir(os.path.join(args.set_dir, set_cur))
        # for im in os.listdir(os.path.join(args.set_dir, set_cur)):
        for k in range(1, len(file_list) + 1):
            if (set_cur == 'LIME'):
                im = str(k) + '.bmp'
            else:
                im = str(k) + '.png'

            if im.endswith(".JPG") or im.endswith(".bmp") or im.endswith(".png"):

                y = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0

                y = cv2.resize(y, (600, 600))
                y_ = torch.from_numpy(np.reshape(y, (1, y.shape[2], y.shape[0], y.shape[1])))
                torch.cuda.synchronize()
                start_time = time.time()
                y_ = y_.cuda()
                x_, _ = model(y_)  # inference
                x_ = x_.view(y.shape[0], y.shape[1], 3)
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                loe_x_ = loe.LOE(ipic=y, epic=x_)

                # print("image: %s LOE = %5.2f" %(im, loe_x_))
                print("image: %s " % im)
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    save_result(x_, path=os.path.join(args.result_dir, model_name, set_cur, im))
                    del x_
                    del y

                    torch.cuda.empty_cache()
                loes.append(loe_x_)

        loe_avg = np.mean(loes)
        loes.append(loe_avg)
        # if args.save_result:
        #     save_result(loe_avg, path=os.path.join(args.result_dir, model_name, set_cur, 'results.txt'))
        # print('LOE=%5.2f', loe_avg)

