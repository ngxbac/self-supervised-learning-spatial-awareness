import numpy as np
import glob
import os
import cv2
import pandas as pd
import SimpleITK
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import SimpleITK
from augmentation import valid_seg_aug
from models import SSUnet


import scipy.ndimage as ndimage


# In the paper
UPPER_BOUND = 400
LOWER_BOUND = -1000


device = torch.device('cuda')


def predict(model, loader):
    model.eval()
    preds = []
    # pred_logits = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = F.softmax(pred, dim=1)
            pred = pred.detach().cpu().numpy()
            # pred = pred.detach().cpu().numpy()
            preds += list(pred)
            # pred_logits.append(pred)

    # preds = np.concatenate(preds, axis=0)
    # pred_logits = np.concatenate(pred_logits, axis=0)
    return preds


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


class TestDataset(Dataset):
    def __init__(self, image_slices, transform):
        self.image_slices = image_slices
        self.transform = transform

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        ymin, ymax, xmin, xmax = bbox(image > 0.5)
        image = image[ymin:350, xmin:xmax]
        image = np.stack((image, image, image), axis=-1).astype(np.float32)

        if self.transform:
            transform = self.transform(image=image)
            image = transform['image']

        image = np.transpose(image, (2, 0, 1))

        return {
            'images': image
        }


# UPPER_BOUND = 400
# LOWER_BOUND = -1000

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


class TestCHAODataset(Dataset):
    def __init__(self, image_slices, transform):
        self.image_slices = image_slices
        self.transform = transform

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image_path = self.image_slices[idx]
        image = load_ct_images(image_path)[0]
        image = norm_image(image)

        # print(image.shape)

        image = np.stack((image, image, image), axis=-1).astype(np.float32)

        if self.transform:
            transform = self.transform(image=image)
            image = transform['image']

        image = np.transpose(image, (2, 0, 1))

        return {
            'images': image
        }


def extract_slice(file):
    ct_image = SimpleITK.ReadImage(file)
    image = SimpleITK.GetArrayFromImage(ct_image).astype(np.float32)

    image = (image - LOWER_BOUND) / (UPPER_BOUND - LOWER_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image.astype(np.float32)

    image_slices = []
    for i, image_slice in enumerate(image):
        image_slices.append(image_slice)

    return image_slices, ct_image


def predict_valid():
    inputdir = "/data/Thoracic_OAR/"
    log_based_dir = "/logs/ss_task3_revise/"
    model_prefix = "Vnet-resnet34-dice7cls-hulung"
    outdir = f"{log_based_dir}/oof_pred/{model_prefix}/"

    transform = valid_seg_aug(image_size=512)
    folds = [0]

    for fold in folds:
        model_name = f"{model_prefix}-fold-{fold}"
        log_dir = f"/logs/ss_task3_revise/{model_name}"

        """
          group_norm: &group_norm False
          classes: 7
          center: !!str &center 'none'
          attention_type: !!str &attention_type 'none'
          reslink: False
          multi_task: &multi_task False
        """
        model = SSUnet(
            encoder_name='resnet34',
            encoder_weights=None,
            classes=7,
        )

        ckp = os.path.join(log_dir, "checkpoints/best.pth")
        checkpoint = torch.load(ckp)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = nn.DataParallel(model)
        model = model.to(device)

        df = pd.read_csv(f'./csv/5folds/valid_{fold}.csv')
        patient_ids = df.patient_id.unique()
        for patient_id in patient_ids:
            nii_file = f"{inputdir}/{patient_id}/data.nii.gz"

            image_slices, ct_image = extract_slice(nii_file)
            dataset = TestDataset(image_slices, transform)
            dataloader = DataLoader(
                dataset=dataset,
                num_workers=4,
                batch_size=8,
                drop_last=False
            )

            pred_mask, pred_logits = predict(model, dataloader)
            # import pdb
            # pdb.set_trace()
            pred_mask = np.argmax(pred_mask, axis=1).astype(np.uint8)
            pred_mask = SimpleITK.GetImageFromArray(pred_mask)

            pred_mask.SetDirection(ct_image.GetDirection())
            pred_mask.SetOrigin(ct_image.GetOrigin())
            pred_mask.SetSpacing(ct_image.GetSpacing())

            # patient_id = nii_file.split("/")[-2]
            patient_dir = f"{outdir}/{patient_id}"
            os.makedirs(patient_dir, exist_ok=True)
            patient_pred = f"{patient_dir}/predict.nii.gz"
            SimpleITK.WriteImage(
                pred_mask, patient_pred
            )


from PIL import Image

def predict_CHAO_CT():
    inputdir = "/data/CHAOS/Train_Sets/CT/"
    transform = valid_seg_aug(image_size=[512, 512])
    log_dir = f"/logs/segmentation/resnet34-pretrained-0/"

    model = SSUnet(
        encoder_name='resnet34',
        encoder_weights=None,
        classes=7,
    )

    ckp = os.path.join(log_dir, "checkpoints/best.pth")
    checkpoint = torch.load(ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)
    model = model.to(device)

    for patient_id in os.listdir("/data/CHAOS/Train_Sets/CT"):
        image_slices = glob.glob(inputdir + f"/{patient_id}/DICOM_anon/*.dcm")
        image_names = [x.split("/")[-1].split(".")[0] for x in image_slices]
        # patient_ids = [x.split("/")[-3] for x in image_slices]

        dataset = TestCHAODataset(image_slices, transform)
        dataloader = DataLoader(
            dataset=dataset,
            num_workers=4,
            batch_size=8,
            drop_last=False
        )

        pred_mask = predict(model, dataloader)
        pred_mask = np.argmax(pred_mask, axis=1).astype(np.uint8)

        os.makedirs("/data/CHAOS/CT_train_predict/", exist_ok=True)
        for pred, image_name in zip(pred_mask, image_names):
            os.makedirs(f"/data/CHAOS/CT_train_predict/{patient_id}", exist_ok=True)
            np.save(f"/data/CHAOS/CT_train_predict/{patient_id}/{image_name}.npy", pred)


if __name__ == '__main__':
    predict_CHAO_CT()
