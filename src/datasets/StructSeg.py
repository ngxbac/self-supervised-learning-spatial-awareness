import numpy as np
import os
import cv2
import glob

import SimpleITK
import pandas as pd
from torch.utils.data import Dataset
from .utils import bbox


class TemporalMixDataset(Dataset):
    def __init__(self, csv_file, transform):
        df = pd.read_csv(csv_file)
        df["slice_idx"] = df["image"].apply(lambda x: int(x.split("/")[-1].split(".")[1]))
        df["max_slice_idx"] = df.groupby("patient_id")["slice_idx"].transform("max")
        df["min_slice_idx"] = df.groupby("patient_id")["slice_idx"].transform("min")
        self.images = df["image"].values
        self.patient_ids = df["patient_id"].values
        self.max_slice_idxs = df["max_slice_idx"].values
        self.min_slice_idxs = df["min_slice_idx"].values
        self.slice_idxs = df["slice_idx"].values

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        patient_id = self.patient_ids[idx]
        max_slice_idx = self.max_slice_idxs[idx]
        min_slice_idx = self.min_slice_idxs[idx]
        slice_idx = self.slice_idxs[idx]

        if slice_idx == min_slice_idx:
            rnd_temporal_idx = 0
        elif slice_idx == min_slice_idx + 1:
            rnd_temporal_idx = np.random.choice([-1, 0], size=1)[0]
        elif slice_idx == max_slice_idx:
            rnd_temporal_idx = 0
        elif slice_idx == max_slice_idx - 1:
            rnd_temporal_idx = np.random.choice([0, 1], size=1)[0]
        else:
            rnd_temporal_idx = np.random.choice([-2, -1, 0, 1, 2], size=1)[0]

        mix_idx = slice_idx + rnd_temporal_idx
        image = np.load(image_path)
        if mix_idx == slice_idx: # No mix
            label = rnd_temporal_idx + 2
            ymin, ymax, xmin, xmax = bbox(image > 0.5)
            image = image[ymin:350, xmin:xmax]
        else:
            mix_image_path = image_path.replace(f".{slice_idx}.", f".{mix_idx}.")
            mix_image = np.load(mix_image_path)

            cutout_size = 50

            ymin, ymax, xmin, xmax = bbox(image > 0.5)
            for i in range(20):
                ystart = np.random.randint(ymin, 350 - cutout_size)
                xstart = np.random.randint(xmin, xmax - cutout_size)
                # mix a part of two images
                image[ystart:ystart + cutout_size, xstart:xstart + cutout_size] = mix_image[ystart:ystart + cutout_size, xstart:xstart + cutout_size]
            image = image[ymin:350, xmin:xmax]
            label = rnd_temporal_idx + 2

        image = np.stack((image, image, image), axis=-1)

        if self.transform:
            image = self.transform(image=image)["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "targets": label
        }


class StructSegTrain2D(Dataset):

    def __init__(self,
                 csv_file,
                 transform
                 ):
        df = pd.read_csv(csv_file)
        self.transform = transform
        self.images = df['image'].values
        self.masks = df['mask'].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]

        # Replace with 2d chest
        # image = image.replace('Thoracic_OAR_2d', 'Thoracic_OAR_2d_lung')
        # mask = mask.replace('Thoracic_OAR_2d', 'Thoracic_OAR_2d_lung')

        image = np.load(image)
        mask = np.load(mask)

        ymin, ymax, xmin, xmax = bbox(image > 0.5)
        image = image[ymin:350, xmin:xmax]
        mask = mask[ymin:350, xmin:xmax]

        image = np.stack((image, image, image), axis=-1).astype(np.float32)

        if self.transform:
            transform = self.transform(image=image, mask=mask)
            image = transform['image']
            mask = transform['mask']

        image = np.transpose(image, (2, 0, 1))
        mask = mask.astype(np.int)

        return {
            'images': image,
            'targets': mask
        }