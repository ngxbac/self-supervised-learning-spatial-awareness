import click
from sklearn.model_selection import KFold, GroupKFold


import numpy as np
import os
import pandas as pd
import SimpleITK
import scipy.ndimage as ndimage
import SimpleITK as sitk

# In the paper
UPPER_BOUND = 400
LOWER_BOUND = -1000

# Look at chest
# UPPER_BOUND = 215
# LOWER_BOUND = -135

# Look at Lungs
# UPPER_BOUND = 150
# LOWER_BOUND = -1350


def load_ct_images(path):
    image = SimpleITK.ReadImage(path)
    image = SimpleITK.GetArrayFromImage(image).astype(np.float32)
    return image


def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(itkimage)


def resize(image, mask):
    image = (image - LOWER_BOUND) / (UPPER_BOUND - LOWER_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image.astype(np.float32)
    return image, mask


def load_patient(imgpath, mskpath):
    image = load_ct_images(imgpath)

    mask = load_ct_images(mskpath)
    image, mask = resize(image, mask)
    return image, mask


def pad_if_need(image, mask, patch):
    assert image.shape == mask.shape

    n_slices, x, y = image.shape
    if n_slices < patch:
        padding = patch - n_slices
        offset = padding // 2
        image = np.pad(image, (offset, patch - n_slices - offset), 'edge')
        mask = np.pad(mask, (offset, patch - n_slices - offset), 'edge')

    return image, mask


def slice_window(image, mask, slice, patch):
    image, mask = pad_if_need(image, mask, patch)
    n_slices, x, y = image.shape
    idx = 0

    image_patches = []
    mask_patches = []

    while idx + patch <= n_slices:
        image_patch = image[idx:idx + patch]
        mask_patch = mask[idx:idx + patch]

        # Save patch
        image_patches.append(image_patch)
        mask_patches.append(mask_patch)

        idx += slice

    return image_patches, mask_patches


def slice_builder(imgpath, mskpath, slice_thichness, scale_ratio, slice, patch, save_dir):
    image, mask = load_patient(imgpath, mskpath, slice_thichness, scale_ratio)
    image_patches, mask_patches = slice_window(image, mask, slice, patch)
    patient_id = imgpath.split("/")[-2]
    save_dir = os.path.join(save_dir, patient_id)
    os.makedirs(save_dir, exist_ok=True)

    image_paths = []
    mask_paths = []
    for i, (image_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
        image_path = os.path.join(save_dir, f'image.{i}.npy')
        mask_path = os.path.join(save_dir, f'mask.{i}.npy')

        image_paths.append(image_path)
        mask_paths.append(mask_path)

        np.save(image_path, image_patch)
        np.save(mask_path, mask_patch)

    df = pd.DataFrame({
        'image': image_paths,
        'mask': mask_paths
    })

    df['patient_id'] = patient_id
    return df


def slice_builder_2d(imgpath, mskpath, save_dir):
    image, mask = load_patient(imgpath, mskpath)
    patient_id = imgpath.split("/")[-2]
    save_dir = os.path.join(save_dir, patient_id)
    os.makedirs(save_dir, exist_ok=True)

    image_paths = []
    mask_paths = []
    for i, (image_slice, mask_slice) in enumerate(zip(image, mask)):
        # if np.any(mask_slice):
        image_path = os.path.join(save_dir, f'image.{i}.npy')
        mask_path = os.path.join(save_dir, f'mask.{i}.npy')

        image_paths.append(image_path)
        mask_paths.append(mask_path)

        np.save(image_path, image_slice)
        np.save(mask_path, mask_slice)

    df = pd.DataFrame({
        'image': image_paths,
        'mask': mask_paths
    })

    df['patient_id'] = patient_id
    return df


def random_crop(image, mask, patch):
    n_slices = image.shape[0]
    start = 0
    end = int(n_slices - patch)

    rnd_idx = np.random.randint(start, end)
    return image[rnd_idx:rnd_idx + patch, :, :], mask[rnd_idx:rnd_idx + patch, :, :]


def center_crop(image, mask, patch):
    n_slices = image.shape[0]
    mid = n_slices // 2
    start = int(mid - patch // 2)
    end = int(mid + patch // 2)

    return image[start:end, :, :], mask[start:end, :, :]


@click.group()
def cli():
    print("Extract slices")


@cli.command()
@click.option('--csv_file', type=str)
@click.option('--root', type=str)
@click.option('--save_dir', type=str)
@click.option('--slice_thichness', type=int)
@click.option('--scale_ratio', type=float)
@click.option('--slice', type=int)
@click.option('--patch', type=int)
def extract(
    csv_file,
    root,
    save_dir,
    slice_thichness,
    scale_ratio,
    slice=16,
    patch=32
):
    df = pd.read_csv(csv_file)
    all_patient_df = []
    for imgpath, mskpath in zip(df.path, df.pathmsk):
        imgpath = os.path.join(root, imgpath)
        mskpath = os.path.join(root, mskpath)
        patient_df = slice_builder(imgpath, mskpath, slice_thichness, scale_ratio, slice, patch, save_dir)
        all_patient_df.append(patient_df)
    all_patient_df = pd.concat(all_patient_df, axis=0).reset_index(drop=True)
    all_patient_df.to_csv(os.path.join(save_dir, 'data.csv'))


@cli.command()
@click.option('--root', type=str)
@click.option('--save_dir', type=str)
def extract_2d(
    root,
    save_dir,
):
    # df = pd.read_csv(csv_file)
    all_patient_df = []
    import glob
    paths = glob.glob(root + "/*/*data*")
    masks = glob.glob(root + "/*/*label*")

    for imgpath, mskpath in zip(paths, masks):
        patient_df = slice_builder_2d(imgpath, mskpath, save_dir)
        all_patient_df.append(patient_df)
    all_patient_df = pd.concat(all_patient_df, axis=0).reset_index(drop=True)
    all_patient_df.to_csv(os.path.join(save_dir, 'data.csv'))


import glob
from tqdm import *

def normalization(volume, axis=None):
    mean = np.mean(volume, axis=axis)
    std = np.std(volume, axis=axis)
    norm_volume = (volume - mean) / std
    return norm_volume


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def get_bbox(image):
    ymin, ymax, xmin, xmax = [], [], [], []
    zero_idx = []
    for idx, i in enumerate(image):
        if i.sum() != 0:
            ymin_, ymax_, xmin_, xmax_ = bbox(i > 0)
            diff1 = ymax_ - ymin_
            diff2 = xmax_ - xmin_
            diff = max(diff1, diff2)
            if diff > 100:
                ymin.append(ymin_)
                ymax.append(ymax_)
                xmin.append(xmin_)
                xmax.append(xmax_)
            else:
                zero_idx.append(idx)
        else:
            zero_idx.append(idx)

    ymin, ymax, xmin, xmax = min(ymin), max(ymax), min(xmin), max(xmax)
    return ymin, ymax, xmin, xmax, zero_idx


def save_slice(des, image):
    for idx, i in enumerate(image):
        np.savez_compressed(os.path.join(des, f"slice_{idx}.npz"), i)


@cli.command()
@click.option('--root', type=str)
@click.option('--save_dir', type=str)
def preprocess_brats19_train(
    root,
    save_dir,
):
    df = pd.read_csv(os.path.join(root, "name_mapping.csv"))
    grades = df["Grade"].values
    ids = df["BraTS_2019_subject_ID"].values
    for grade, id in tqdm(zip(grades, ids), total=len(grades)):
        os.makedirs(os.path.join(save_dir, grade, id), exist_ok=True)

        images = []

        file = os.path.join(root, grade, id, id + "_flair.nii.gz")
        image = load_ct_images(file)
        ymin, ymax, xmin, xmax, zero_idx = get_bbox(image)
        keep_idx = list(set(range(image.shape[0])) - set(zero_idx))
        # image = image[:, ymin:ymax, xmin:xmax]
        # image = image[keep_idx]
        # image = normalization(image)
        images.append(image)

        for modal in ["_t1", "_t2", "_t1ce"]:
            file = os.path.join(root, grade, id, id + f"{modal}.nii.gz")
            image = load_ct_images(file)
            # image = image[:, ymin:ymax, xmin:xmax]
            # image = image[keep_idx]
            # image = normalization(image)
            images.append(image)

        images = np.asarray(images)
        images = np.transpose(images, (1, 2, 3, 0))

        path = os.path.join(save_dir, grade, id, "images")
        os.makedirs(path, exist_ok=True)
        save_slice(path, images)

        # Mask
        file = os.path.join(root, grade, id, id + "_seg.nii.gz")
        image = load_ct_images(file)
        # image = image[:, ymin:ymax, xmin:xmax]
        # image = image[keep_idx]
        # image = normalization(image)
        path = os.path.join(save_dir, grade, id, "masks")
        os.makedirs(path, exist_ok=True)
        save_slice(path, image)


@cli.command()
@click.option('--root', type=str)
@click.option('--save_dir', type=str)
def preprocess_brats19_valid(
    root,
    save_dir,
):
    df = pd.read_csv(os.path.join(root, "name_mapping_validation_data.csv"))
    # grades = df["Grade"].values
    ids = df["BraTS_2019_subject_ID"].values
    for id in tqdm(ids, total=len(ids)):
        os.makedirs(os.path.join(save_dir, id), exist_ok=True)

        file = os.path.join(root, id, id + "_flair.nii.gz")
        if not os.path.isfile(file):
            print(file)
            continue
        image = load_ct_images(file)
        ymin, ymax, xmin, xmax, zero_idx = get_bbox(image)
        keep_idx = list(set(range(image.shape[0])) - set(zero_idx))
        image = image[:, ymin:ymax, xmin:xmax]
        image = image[keep_idx]
        image = normalization(image)

        path = os.path.join(save_dir, id, id + "_flair")
        os.makedirs(path, exist_ok=True)
        save_slice(path, image)

        for modal in ["_t1", "_t2", "_t1ce", "_seg"]:
            file = os.path.join(root, id, id + f"{modal}.nii.gz")
            image = load_ct_images(file)
            image = image[:, ymin:ymax, xmin:xmax]
            image = image[keep_idx]
            if modal != "_seg":
                image = normalization(image)

            path = os.path.join(save_dir, id, id + f"{modal}")
            os.makedirs(path, exist_ok=True)
            save_slice(path, image)



@cli.command()
@click.option('--csv_file', type=str)
@click.option('--n_folds', type=int)
@click.option('--save_dir', type=str)
def split_kfold(
    csv_file,
    n_folds,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    patient_ids = df['patient_id'].values
    kf = GroupKFold(n_splits=n_folds)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(df, groups=patient_ids)):
        # train_patient = patient_ids[train_idx]
        # valid_patient = patient_ids[valid_idx]
        # train_df = df[df['patient_id'].isin(train_patient)].reset_index(drop=True)
        # valid_df = df[df['patient_id'].isin(valid_patient)].reset_index(drop=True)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        train_df.to_csv(os.path.join(save_dir, f'train_{fold}.csv'), index=False)
        valid_df.to_csv(os.path.join(save_dir, f'valid_{fold}.csv'), index=False)


@cli.command()
@click.option('--csv_file', type=str)
@click.option('--n_folds', type=int)
@click.option('--save_dir', type=str)
def split_kfold_proxy_task(
    csv_file,
    n_folds,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    patient_ids = df['patient_id'].values
    kf = GroupKFold(n_splits=n_folds)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(df, groups=patient_ids)):
        # train_patient = patient_ids[train_idx]
        # valid_patient = patient_ids[valid_idx]
        # train_df = df[df['patient_id'].isin(train_patient)].reset_index(drop=True)
        # valid_df = df[df['patient_id'].isin(valid_patient)].reset_index(drop=True)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        train_df.to_csv(os.path.join(save_dir, f'train_{fold}.csv'), index=False)
        valid_df.to_csv(os.path.join(save_dir, f'valid_{fold}.csv'), index=False)


@cli.command()
@click.option('--csv_file', type=str)
@click.option('--n_folds', type=int)
@click.option('--save_dir', type=str)
def split_kfold_semi(
    csv_file,
    n_folds,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    all_patients = df['patient_id'].unique()
    unlabeled_patients = np.random.choice(all_patients, size=10, replace=False)
    unlabeled_df = df[df['patient_id'].isin(unlabeled_patients)]
    labeled_df = df[~df['patient_id'].isin(unlabeled_patients)].reset_index(drop=True)
    patient_ids = labeled_df['patient_id'].values
    kf = GroupKFold(n_splits=n_folds)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(labeled_df, groups=patient_ids)):
        train_df = labeled_df.iloc[train_idx].reset_index(drop=True)
        valid_df = labeled_df.iloc[valid_idx].reset_index(drop=True)

        train_df.to_csv(os.path.join(save_dir, f'train_{fold}.csv'), index=False)
        valid_df.to_csv(os.path.join(save_dir, f'valid_{fold}.csv'), index=False)

    unlabeled_df.to_csv(os.path.join(save_dir, 'unlabeled_patients.csv'), index=False)


if __name__ == '__main__':
    cli()
