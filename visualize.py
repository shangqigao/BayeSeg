import torchvision.transforms.functional as F
import torch.nn.functional as Func
import torchvision.transforms as T
import math
import sys
import random
import time
import datetime
import tqdm
from typing import Iterable
import numpy as np
import PIL 
from PIL import Image
from skimage import transform
import nibabel as nib
import torch
import os
from medpy.metric.binary import dc
import pandas as pd
import glob
import re
import shutil
import copy
from skimage import measure
import matplotlib.pyplot as plt

import util.misc as utils


def makefolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def save_cuda_img(folder, name, outputs, index=0, log=None):
    img = outputs[name]
    img = img.detach().cpu().numpy()
    img = img[0, index, ...]
    img = img[::-1, :]
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    if log is not None:
        img = np.log(1 + log * img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    img = (img * 255).astype(np.uint8)
    save_path = os.path.join(folder, name + '.png')
    plt.imsave(save_path, img, cmap='gray')
    

def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def conv_int(i):
    return int(i) if i.isdigit() else i

def natural_order(sord):
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    # keep a heart connectivity 
    mask_shape = mask.shape
    
    heart_slice = np.where((mask>0), 1, 0)
    out_heart = np.zeros(heart_slice.shape, dtype=np.uint8)
    for struc_id in [1]:
        binary_img = heart_slice == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_heart[blobs == largest_blob_label] = struc_id
    
    # keep LV/RV/MYO connectivity
    # out_img = np.zeros(mask.shape, dtype=np.uint8)
    # for struc_id in [1, 2, 3]:
    #     binary_img = out_heart == struc_id
    #     blobs = measure.label(binary_img, connectivity=1)
    #     props = measure.regionprops(blobs)
    #     if not props:
    #         continue
    #     area = [ele.area for ele in props]
    #     largest_blob_ind = np.argmax(area)
    #     largest_blob_label = props[largest_blob_ind].label
    #     out_img[blobs == largest_blob_label] = struc_id
    #final_img = out_img
    final_img = out_heart * mask
    return final_img

@torch.no_grad()
def visual(model, model_type, dataloader_dict, output_folder, device):
    model.eval()
    #criterion.eval()
    
    dataset = 'MSCMR'
    if dataset == 'MSCMR':
        test_folder = "../Datasets/MSCMR_dataset/test/T2/images/"
        label_folder = "../Datasets/MSCMR_dataset/test/T2/labels/"
    elif dataset == 'ACDC':
        test_folder = "../nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC/imagesTr" 
        label_folder = "../nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC/labelsTr"
    else:
        raise ValueError('Invalid dataset: {}'.format(dataset))

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    makefolder(output_folder)

    target_resolution = (1.36719, 1.36719)

    test_file = 'patient43_T2.nii.gz'
    slice_index = 4
    # read_image
    img_path = os.path.join(test_folder, test_file)
    img_dat = load_nii(img_path)
    img = img_dat[0].copy()

    pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
    scale_vector = (pixel_size[0] / target_resolution[0],
                    pixel_size[1] / target_resolution[1])

    
    img = img.astype(np.float32)
    img = np.divide((img - np.mean(img)), np.std(img))
    
    print(img.shape, pixel_size)
    img_slice = np.squeeze(img[:,:,slice_index])
    slice_rescaled = transform.rescale(img_slice,
                                    scale_vector,
                                    order=1,
                                    preserve_range=True,
                                    multichannel=False,
                                    anti_aliasing=True,
                                    mode='constant')
    img_slice = slice_rescaled
    nx = 212
    ny = 212
    x, y = img_slice.shape
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2
    # Crop section of image for prediction
    if x > nx and y > ny:
        slice_cropped = img_slice[x_s:x_s+nx, y_s:y_s+ny]
    else:
        slice_cropped = np.zeros((nx,ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c+ x, :] = img_slice[:,y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c+x, y_c:y_c + y] = img_slice[:, :]
    
    img_slice = slice_cropped
    img_slice = np.divide((slice_cropped - np.mean(slice_cropped)), np.std(slice_cropped))
    img_slice = np.reshape(img_slice, (1,1,nx,ny))

    img_slice = torch.from_numpy(img_slice)
    img_slice = img_slice.to(device)
    img_slice = img_slice.float()
    
    tasks = dataloader_dict.keys()
    task = random.sample(tasks, 1)[0]
   
    if model_type == 'BayeSeg' or model_type == 'Unet':
        outputs = model(img_slice, task)
    elif model_type == 'PUnet':
        outputs = model(img_slice, task, training=False)
    else:
        return ValueError('Invalid model: {}'.format(model_type))
   
    outputs = outputs['visualize'] 
    save_cuda_img(output_folder, 'y', outputs, index=0)
    save_cuda_img(output_folder, 'n', outputs, index=0)
    save_cuda_img(output_folder, 'm', outputs, index=0)
    save_cuda_img(output_folder, 'rho', outputs, index=0)
    save_cuda_img(output_folder, 'x', outputs, index=0, log=1e1)
    save_cuda_img(output_folder, 'upsilon', outputs, index=0, log=1e1)
    save_cuda_img(output_folder, 'z', outputs, index=2)
    save_cuda_img(output_folder, 'omega', outputs, index=2)
    return outputs

