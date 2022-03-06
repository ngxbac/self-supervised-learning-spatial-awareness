from albumentations import *
import cv2
import random
import numpy as np
from albumentations.pytorch import ToTensorV2


def train_aug(image_size):
    return Compose([
        Resize(*image_size),
        HorizontalFlip(),
        Normalize(max_pixel_value=1),
    ], p=1)


def valid_aug(image_size):
    return Compose([
        Resize(*image_size),
        Normalize(max_pixel_value=1),
    ], p=1)


def train_seg_aug(image_size):
    return Compose([
        Resize(*image_size),
        Normalize(max_pixel_value=1),
        # ToTensorV2(),
    ], p=1)


def train_brats_aug(image_size):
    return Compose([
        Resize(*image_size),
        HorizontalFlip(),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15),
        # Normalize(max_pixel_value=1),
        # ToTensorV2(),
    ], p=1)


def valid_seg_aug(image_size):
    return Compose([
        Resize(*image_size),
        Normalize(max_pixel_value=1),
        # ToTensorV2(),
    ], p=1)


def valid_brats_aug(image_size):
    return Compose([
        Resize(*image_size),
        # Normalize(max_pixel_value=1),
        # ToTensorV2(),
    ], p=1)
