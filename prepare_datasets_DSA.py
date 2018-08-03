#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import cv2
from skimage import io

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_train= "./DSA/inputs/"
groundTruth_train1="./DSA/targets1/"
groundTruth_train2="./DSA/targets2/"


#original_imgs_train = "./DRIVE/training/images/"
#groundTruth_imgs_train = "./DRIVE/training/1st_manual/"
#borderMasks_imgs_train = "./DRIVE/training/mask/"

#test
original_test="./DSA/test_data/"
groundTruth_test1="./DSA/test_targets1/"
groundTruth_test2="./DSA/test_targets2/"

#original_imgs_test = "./DRIVE/test/images/"
#groundTruth_imgs_test = "./DRIVE/test/1st_manual/"
#borderMasks_imgs_test = "./DRIVE/test/mask/"
#---------------------------------------------------------------------------------------------

Nimgs = 96
channels = 1
height = 1024
width = 1024
dataset_path = "./DSA_datasets_training_testing/"


def get_datasets(imgs_dir,groundTruth_dir1,groundTruth_dir2,train_test="null"):
    if train_test=="test":
        Nimgs=32
    else:
        Nimgs=96
    
    inputs = []
    groundTruth=[]
    files=os.listdir(imgs_dir)
    files1=os.listdir(groundTruth_dir1)
    files2=os.listdir(groundTruth_dir2)
   # border_masks = np.empty((Nimgs,height,width))
       
    for file in files:
         input_file=os.path.join(imgs_dir,file)
         img = cv2.imread(input_file,0)
         inputs.append(img)
         if os.path.exists(os.path.join(groundTruth_dir1,file)):
            target_file=os.path.join(groundTruth_dir1,file)
            target_image=np.array(io.imread(target_file))
            target_image=cv2.threshold(target_image, 127, 1, cv2.THRESH_BINARY)[1]
            groundTruth.append(target_image)         
         elif os.path.exists(os.path.join(groundTruth_dir2,file)):
            target_file=os.path.join(groundTruth_dir2,file)
            target_image=np.array(io.imread(target_file))[:,:,3]
            target_image=cv2.threshold(target_image,127,1,cv2.THRESH_BINARY)[1]
            groundTruth.append(target_image)
         # b_mask = Image.open(borderMasks_dir + border_masks_name)
           # border_masks[i] = np.asarray(b_mask)
    inputs=np.asarray(inputs)
    inputs=np.expand_dims(inputs,axis=3)
    groundTruth=np.asarray(groundTruth)
    print (inputs.shape)
    print(groundTruth.shape)

    print ("imgs max: " +str(np.max(inputs)))
    print ("imgs min: " +str(np.min(inputs)))
    assert(np.max(groundTruth)==1 )
    assert(np.min(groundTruth)==0 )
    print ("ground truth are correctly withih pixel value range 0-1 (black-white)")
    #reshaping for my standard tensors
    inputs = np.transpose(inputs,(0,3,1,2))
    assert(inputs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
   # border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs, 1,height,width))
    #assert(border_masks.shape == (Nimgs,1,height,width))
    return inputs, groundTruth

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
imgs_train, groundTruth_train = get_datasets(original_train,groundTruth_train1,groundTruth_train2,"train")
print ("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DSA_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DSA_dataset_groundTruth_train.hdf5")


#getting the testing datasets
imgs_test, groundTruth_test = get_datasets(original_test,groundTruth_test1,groundTruth_test2,"test")
print ("saving test datasets")
write_hdf5(imgs_test,dataset_path + "DSA_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DSA_dataset_groundTruth_test.hdf5")
