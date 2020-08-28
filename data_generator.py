# based on the code from https://github.com/cszn/DnCNN

# from __future__ import print_function
import glob, os
import cv2
import numpy as np
from torch.utils.data import Dataset
from config import *
import random
import os
import scipy.io as sio


patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]

batch_size = BATCH_SIZE


class LightingDataset(Dataset):

    def __init__(self, xs, illums):
        super(LightingDataset, self).__init__()
        self.xs = xs
        self.illums = illums

    def __getitem__(self, index):
        batch_x = self.xs[index]
        batch_y = batch_x * (self.illums[index])

        return batch_x, batch_y, self.illums[index]

    def __len__(self):
        return self.xs.size(0)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name)  # 3 ch

    if (img is not None):

        img = cv2.resize(img, (180, 180))  # adj dims
        h, w, ch = img.shape

        patches = []
        illums = []
        for s in scales:
            h_scaled, w_scaled = int(h * s), int(w * s)
            img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            # extract patches
            for i in range(0, h_scaled - patch_size + 1, stride):
                for j in range(0, w_scaled - patch_size + 1, stride):
                    x = img_scaled[i:i + patch_size, j:j + patch_size, :]  # 3 ch
                    for k in range(0, aug_times):
                        x_aug = data_aug(x, mode=np.random.randint(0, 8))
                        patches.append(x_aug)
                        illums.append(random.uniform(0.1, 0.5)) # illuminacao

        return patches, illums
    else:
        print('Image not found:', file_name)
        return [], []


def datagenerator(data_dir='data/Train400', verbose=False, batch_size=50, is_validation=False):
    if is_validation:
        random.seed(0)
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir + '/*.jpg')  # get name list of all .jpg files

    # initrialize
    data = []
    data_illu = []
    # generate patches
    print('data generatooor: ', len(file_list), data_dir)
    for i in range(len(file_list)):
        patches, illums = gen_patches(file_list[i])

        if (len(patches) != 0):
            for i in range(len(patches)):
                data.append(patches[i])
                data_illu.append(illums[i])

        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype=np.uint8)
    data_illu = np.array(data_illu)
    discard_n = len(data) - len(data) // batch_size * batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    data_illu = np.delete(data_illu, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')

    return data, data_illu


# if __name__ == '__main__':
#     data, illums = datagenerator(data_dir=DATASET_DIR + '/original', batch_size=128)
#     print('salvando dados...')
#     sio.savemat(os.path.join(DATASET_DIR, 'xs.mat'), {'data': data, 'illums': illums})
#     print('end \o')

