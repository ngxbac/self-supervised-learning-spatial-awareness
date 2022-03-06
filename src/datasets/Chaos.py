import numpy as np
import os
import cv2
import glob

import SimpleITK
import pandas as pd
from torch.utils.data import Dataset
from .utils import bbox, LOWER_BOUND, UPPER_BOUND


def norm_image(image):
    image = (image - LOWER_BOUND) / (UPPER_BOUND - LOWER_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image.astype(np.float32)
    return image


def load_ct_images(path):
    image = SimpleITK.ReadImage(path)
    image = SimpleITK.GetArrayFromImage(image).astype(np.float32)
    return image


class CHAOSDataset(Dataset):

    def __init__(self,
                 ct_dir,
                 pred_dir,
                 transform
                 ):

        ct_files = glob.glob(ct_dir + "/*/DICOM_anon/*.dcm")
        patient_ids = [x.split("/")[-3] for x in ct_files]
        file_names = [x.split("/")[-1].split(".")[0] for x in ct_files]
        gt_files = [
            pred_dir + f"/{patient_id}/{filename}.npy" for patient_id, filename in zip(patient_ids, file_names)
        ]

        self.transform = transform
        self.images = ct_files
        self.targets = gt_files
        # print(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print(idx)

        image = self.images[idx]
        image = load_ct_images(image)[0]
        image = norm_image(image)

        ymin, ymax, xmin, xmax = bbox(image > 0.5)
        image = image[ymin:ymax, xmin:xmax]
        # mask = mask[ymin:350, xmin:xmax]

        target = self.targets[idx]
        target = np.load(target)
        labels = np.unique(target)
        target = np.zeros((7, ), dtype=np.float32)
        for label in labels:
            target[label] = 1

        # print(target)

        image = np.stack((image, image, image), axis=-1).astype(np.float32)

        if self.transform:
            transform = self.transform(image=image)
            image = transform['image']

        image = np.transpose(image, (2, 0, 1))

        return {
            'images': image,
            'targets': target
        }