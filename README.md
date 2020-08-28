# Enhancement of Weakly Illuminated Images Using CNN and Retinex Theory (SMC 2020)

a) **Dependencies**
 - Pytorch
 - OpenCV
 - Skimage
 - Numpy
 
b) **Datasets**
 - all datasets used in training and testing are compressed in the /data folder.

c) **Training and testing**

- train:
`python main_train.py --model`


- testing dataset of real scenes:
`python main_test.py --model_dir --set_names`


- testing synthetic dataset:
`python main_synthetic_test.py --model_dir --set_names`

d) **Metrics**
- PSNR and SSIM (from skimage.measure)
- [LOE](https://github.com/baidut/BIMEF/blob/master/quality/loe100x100.m)