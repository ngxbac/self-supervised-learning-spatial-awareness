import numpy as np


UPPER_BOUND = 400
LOWER_BOUND = -1000


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop(image):
    ymin,ymax,xmin,xmax = bbox(image > 0.5)
    return image[ymin:ymax, xmin:xmax]


def normalization(volume, axis=None):
    mean = np.mean(volume, axis=axis)
    std = np.std(volume, axis=axis)
    norm_volume = (volume - mean) / std
    return norm_volume
