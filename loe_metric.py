import numpy as np

def LOE(ipic=None, epic=None):
    # original image
    m, n, k = ipic.shape


    #get the local maximum for each pixel of the input image
    # win = 7
    win = 100
    # imax = round(max(max(ipic(:,:, 1), ipic(:,:, 2)), ipic(:,:, 3)));

    imax = np.round(np.maximum(np.maximum(ipic[:,:,0],ipic[:,:,1]), ipic[:,:,2]))
    imax = getlocalmax(imax, win)
    #get the local maximum for each pixel of the enhanced image
    emax = np.round(np.maximum(np.maximum(epic[:,:,0],epic[:,:,1]), epic[:,:,2]))
    emax = getlocalmax(emax, win)

    #get the downsampled image
    blkwin = 50
    mind = min(m, n)
    step = int(np.floor(mind / blkwin))# the step to down sample the image
    blkm = int(np.floor(m / step))
    blkn = int(np.floor(n / step))
    ipic_ds = np.zeros(shape=(blkm, blkn))# downsampled of the input image
    epic_ds = np.zeros(shape=(blkm, blkn))# downsampled of the enhanced image
    LOE = np.zeros(shape=(blkm, blkn))#

    for i in range(blkm):
        for j in range(blkn):
            ipic_ds[i, j] = imax[i * step, j * step]
            epic_ds[i, j] = emax[i * step, j * step]


    for i in range(blkm):
        for j in range(blkn):        #bug
            flag1 = ipic_ds >= ipic_ds[i, j]
            flag2 = epic_ds >= epic_ds[i, j]
            flag = (flag1 != flag2)
            flag_aux = flag.astype("uint8")
            LOE[i, j] = sum(flag_aux.flatten())

    #         LOE=mean(LOE(:))
    value = np.mean(LOE[:])
    return value

# # @mfunction("output")
def getlocalmax(pic=None, win=None):

    m, n = pic.shape
    extpic = getextpic(pic, win)
    output = np.zeros(shape=(m, n))

    for i in range(1+win, m + win):
        for j in range(1+win, n + win):
            modual = extpic[i - win:i + win, j - win:j + win]
            output[i - win, j - win] = np.max(modual[:])

    return output


# @mfunction("output")
def getextpic(im=None, win_size=None):

    h, w = im.shape

    extpic = np.zeros(shape=(h + 2 * win_size, w + 2 * win_size))
    extpic[win_size:win_size + h, win_size:win_size + w] = np.copy(im)
    # extpic(win_size + 1:win_size + h, win_size + 1:win_size + w,:)=im;
    for i in range(win_size):    #extense row
        extpic[win_size+1-i,win_size+1:win_size+w] = extpic[win_size+1+i,win_size+1:win_size+w]   #top edge
        extpic[h + win_size + i, win_size + 1:win_size + w]=extpic[h + win_size - i, win_size + 1:win_size + w]

    for i in range(win_size):   #extense column
        extpic[:, win_size + 1 - i] = extpic[:, win_size + 1 + i] # left edge
        extpic[:, win_size + w + i] = extpic[:, win_size + w - i] # right edge


    return extpic
