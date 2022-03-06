import numpy as np
import os
import cv2
import glob

import SimpleITK
import pandas as pd
from torch.utils.data import Dataset
from .utils import bbox, normalization


class Brats2019Dataset(Dataset):

    def __init__(self,
                 root_dir,
                 csv_file,
                 transform,
                 data="WT"
                 ):

        df = pd.read_csv(csv_file)
        self.images = glob.glob(root_dir + f"/*/*/images/*.npz")
        self.images = [image for image in self.images if image.split("/")[-3] in df["BraTS_2019_subject_ID"].values]
        self.transform = transform
        self.data = data
        print(self.data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = image.replace("images", "masks")

        image = np.load(image)["arr_0"]#"[:, :, 0]
        ymin, ymax, xmin, xmax = bbox(image > 0)

        mask = np.load(mask)["arr_0"]
        # print("Before ", np.unique(mask))
        label1 = np.where(mask == 1)
        label2 = np.where(mask == 2)
        label3 = np.where(mask == 4)

        if self.data == "WT":
            mask[label1] = 1
            mask[label2] = 1
            mask[label3] = 1
        elif self.data == "TC":
            mask[label1] = 1
            mask[label2] = 0
            mask[label3] = 1
        elif self.data == "E":
            mask[label1] = 0
            mask[label2] = 0
            mask[label3] = 1
        # mask = np.expand_dims(mask, axis=0)
        # print(mask.shape)
        # print("After ", np.unique(mask))

        # image = image / image.max()
        image = normalization(image)
        image = image[ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]
        # image = np.stack((image, image, image), axis=-1)

        # image = np.stack((image, image, image), axis=-1).astype(np.float32)
        # image = np.transpose(image, (1, 2, 0)).astype(np.float32)

        if self.transform:
            transform = self.transform(image=image, mask=mask)
            image = transform['image']
            mask = transform["mask"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        # mask = mask.astype(np.int)

        return {
            'images': image,
            'targets': mask
        }


class BratsTemporalMixDataset(Brats2019Dataset):
    def __init__(self,
                 root_dir,
                 csv_file,
                 transform,
                 ):
        super(BratsTemporalMixDataset, self).__init__(
            root_dir,
            csv_file,
            transform
        )

        df = pd.DataFrame({
            "image": self.images
        })
        df["patient_id"] = df["image"].apply(lambda x: x.split("/")[-3])
        df["slice_idx"] = df["image"].apply(lambda x: int(x.split("/")[-1].split(".")[0].split("_")[1]))
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

        # print(slice_idx, min_slice_idx, max_slice_idx)

        if min_slice_idx <= slice_idx < min_slice_idx + 3:
            rnd_temporal_idx = np.random.choice([0, 3, 5], size=1)[0]
        elif min_slice_idx + 3 <= slice_idx < min_slice_idx + 5:
            rnd_temporal_idx = np.random.choice([-3, 0, 3, 5], size=1)[0]
        elif min_slice_idx + 5 <= slice_idx <= max_slice_idx - 5:
            rnd_temporal_idx = np.random.choice([-5, -3, 0, 3, 5], size=1)[0]
        elif max_slice_idx - 5 < slice_idx <= max_slice_idx - 3:
            rnd_temporal_idx = np.random.choice([-5, -3, 0, 3], size=1)[0]
        elif slice_idx > max_slice_idx - 3:
            rnd_temporal_idx = np.random.choice([-5, -3, 0], size=1)[0]
        else:
            print(slice_idx)

        mix_idx = slice_idx + rnd_temporal_idx
        image = np.load(image_path)["arr_0"]
        ymin, ymax, xmin, xmax = bbox(image > 0)
        diff1 = ymax - ymin
        diff2 = xmax - xmin
        if diff1 > 100 and diff2 > 100:
            if mix_idx == slice_idx:  # No mix
                label = 0
            else:
                mix_image_path = image_path.replace(f"_{slice_idx}.", f"_{mix_idx}.")
                mix_image = np.load(mix_image_path)["arr_0"]

                cutout_size = 20
                for i in range(20):
                    ystart = np.random.randint(ymin, ymax - cutout_size)
                    xstart = np.random.randint(xmin, xmax - cutout_size)
                    # mix a part of two images
                    image[ystart:ystart + cutout_size, xstart:xstart + cutout_size] = mix_image[
                                                                                      ystart:ystart + cutout_size,
                                                                                      xstart:xstart + cutout_size]
                    # image = cv2.rectangle(
                    #     image,
                    #     (ystart, xstart),
                    #     (ystart + cutout_size, xstart + cutout_size),
                    #     thickness=1,
                    #     color=(255, 255, 0)
                    # )
                # image = image[ymin:350, xmin:xmax]
                # label = rnd_temporal_idx + 2
                if rnd_temporal_idx == -5:
                    label = 0
                elif rnd_temporal_idx == -3:
                    label = 1
                elif rnd_temporal_idx == 0:
                    label = 2
                elif rnd_temporal_idx == 3:
                    label = 3
                elif rnd_temporal_idx == 5:
                    label = 4
                else:
                    raise ValueError("Invalid valude")
        else:
            label = 2  # No mix

        image = normalization(image)
        image = image[ymin:ymax, xmin:xmax]

        if self.transform:
            image = self.transform(image=image)["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "targets": label
        }