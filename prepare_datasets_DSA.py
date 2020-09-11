# ==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
# ============================================================

import os
import h5py
import numpy as np
from PIL import Image
import cv2
from skimage import io


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# ------------Path of the images --------------------------------------------------------------
# train
original_train = '/root/data/vessels/train/images'
groundTruth_train = '/root/data/vessels/train/gt'

# test
original_test = '/root/data/vessels/test/images'
groundTruth_test = '/root/data/vessels/test/gt'

# ---------------------------------------------------------------------------------------------

Nimgs = 96
channels = 1
height = 1024
width = 1024
dataset_path = "./DSA_datasets_training_testing/"


def get_datasets(imgs_dir, groundTruth_dir, train_test="null"):
    if train_test == "test":
        Nimgs = 32
    else:
        Nimgs = 96

    inputs = []
    groundTruth = []
    files = os.listdir(imgs_dir)

    for file in files:
        input_file = os.path.join(imgs_dir, file)
        img = cv2.imread(input_file, 0)
        inputs.append(img)

        mask_file = os.path.join(groundTruth_dir, file+".npy")
        mask = np.load(mask_file)
        groundTruth.append(mask)

    inputs = np.asarray(inputs)
    inputs = np.expand_dims(inputs, axis=3)
    groundTruth = np.asarray(groundTruth)

    print(inputs.shape)
    print(groundTruth.shape)

    print("imgs max: " + str(np.max(inputs)))
    print("imgs min: " + str(np.min(inputs)))
    assert (np.max(groundTruth) == 1)
    assert (np.min(groundTruth) == 0)
    print("ground truth are correctly withih pixel value range 0-1 (black-white)")

    # reshaping for my standard tensors
    inputs = np.transpose(inputs, (0, 3, 1, 2))
    assert (inputs.shape == (Nimgs, channels, height, width))
    groundTruth = np.reshape(groundTruth, (Nimgs, 1, height, width))

    assert (groundTruth.shape == (Nimgs, 1, height, width))
    return inputs, groundTruth


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# getting the training datasets
imgs_train, groundTruth_train = get_datasets(original_train, groundTruth_train, "train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DSA_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DSA_dataset_groundTruth_train.hdf5")

# getting the testing datasets
imgs_test, groundTruth_test = get_datasets(original_test, groundTruth_test, "test")
print("saving test datasets")
write_hdf5(imgs_test, dataset_path + "DSA_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DSA_dataset_groundTruth_test.hdf5")
