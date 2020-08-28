import os
import cv2
import numpy as np
from skimage import io
from os import listdir
from os.path import isfile, join


def get_mean_stDeviation(img_names_list):
    average_color = np.array([0., 0., 0.])
    std_color = np.array([0., 0., 0.])
    for k in range(0, len(img_names_list)):
        img_name = img_names_list[k]
        img = io.imread(img_name)
        average_color_local = [img[:, :, i].mean() for i in range(img.shape[-1])]
        average_color = np.add(average_color, average_color_local)
        std_color = np.add(std_color, [img[:, :, j].std() for j in range(img.shape[-1])])

    total_images = len(img_names_list)
    average_color = np.asarray(average_color) / [255.0 * total_images, 255.0 * total_images, 255.0 * total_images]
    std_color = np.asarray(std_color) / [255.0 * total_images, 255.0 * total_images, 255.0 * total_images]
    return average_color, std_color


# def standardize_filenames(filenames_dir):
#     images = sorted([f for f in listdir(filenames_dir) if isfile(join(filenames_dir, f))])
#
#     for i in range(len(images)):
#         print (os.path.join(filenames_dir, images[i]))
#         os.rename(os.path.join(filenames_dir, images[i]), os.path.join(filenames_dir, str(i + 1) + '.png'))
#     print(images)


def blockshaped(arr, nrows, ncols):
    """ <https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays/16858283#16858283>
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    if (len(arr.shape) == 3):
        h, w, ch = arr.shape
    else:
        h, w = arr.shape
        ch = 1
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h // nrows, nrows, -1, ncols, ch)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols, ch))


def recompose_image(arr_patches, arr_illu):
    n_patches, h_patch, w_patch, ch = arr_patches.shape

    image_size = int(np.sqrt(n_patches) * h_patch)  # 50*10 = 500
    restaured_image = np.zeros((image_size, image_size, ch))
    illu_map = np.zeros((image_size, image_size))
    patch_size = h_patch

    k = 0
    for i in range(0, image_size - patch_size + 1, patch_size):
        for j in range(0, image_size - patch_size + 1, patch_size):
            restaured_image[i:i + patch_size, j:j + patch_size, :] = arr_patches[k]
            illu_map[i:i + patch_size, j:j + patch_size] = np.asarray(
                [[arr_illu[k] for r in range(patch_size)] for c in range(patch_size)])

            k += 1
    return restaured_image, illu_map


def smooth_illu_map(illu_map):

    smooth_map = np.copy(illu_map)
    size = 100
    kernel = np.ones((size, size), np.float32) / (size ** 2)
    smooth_map = cv2.filter2D(smooth_map, -1, kernel)

    return smooth_map
# if __name__ == '__main__':
#     filenames = standardize_filenames('data/test/new_synthetic_kodak')