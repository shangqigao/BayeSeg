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
def infer(model, model_type, dataset, sequence, dataloader_dict, output_folder, device):
    model.eval()
    #criterion.eval()
    
    #dataset = 'MSCMR'
    if dataset in ['MSCMR', 'ACDC']:
        test_folder = "../Datasets/{}/test/{}/images/".format(dataset, sequence)
        label_folder = "../Datasets/{}/test/{}/labels/".format(dataset, sequence)
    else:
        raise ValueError('Invalid dataset: {}'.format(dataset))
    
    output_folder = os.path.join(output_folder, dataset, 'test', str(model_type), sequence)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    makefolder(output_folder)

    target_resolution = (1.36719, 1.36719)

    test_files = sorted(os.listdir(test_folder))
    label_files = sorted(os.listdir(label_folder))
    assert len(test_files) == len(label_files)

    # read_image
    for file_index in range(len(test_files)):
        test_file = test_files[file_index] 
        print('Processing {}'.format(test_file))        

        label_file = label_files[file_index]
        file_mask = os.path.join(label_folder, label_file)
        mask_dat = load_nii(file_mask)
        mask = mask_dat[0]

        img_path = os.path.join(test_folder, test_file)
        img_dat = load_nii(img_path)
        img = img_dat[0].copy()

        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])

        
        img = img.astype(np.float32)
        img = np.divide((img - np.mean(img)), np.std(img))
        
        # print(img.shape, pixel_size)
        slice_rescaleds = []
        for slice_index in range(img.shape[2]):
            img_slice = np.squeeze(img[:,:,slice_index])
            slice_rescaled = transform.rescale(img_slice,
                                            scale_vector,
                                            order=1,
                                            preserve_range=True,
                                            multichannel=False,
                                            anti_aliasing=True,
                                            mode='constant')
            slice_rescaleds.append(slice_rescaled)
        img = np.stack(slice_rescaleds, axis=2)

        predictions = []

        for slice_index in range(img.shape[2]):
            img_slice = img[:,:,slice_index]
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
          
            if model_type in ['BayeSeg', 'Unet', 'Baseline']:
                outputs = model(img_slice, task)
            elif model_type == 'PUnet':
                outputs = model(img_slice, None, training=False)
            else:
                return ValueError('Invalid model: {}'.format(model_type))
            
            softmax_out = outputs["pred_masks"]
            softmax_out = softmax_out.detach().cpu().numpy()
            prediction_cropped = np.squeeze(softmax_out[0,...])

            slice_predictions = np.zeros((4,x,y))
            # insert cropped region into original image again
            if x > nx and y > ny:
                slice_predictions[:,x_s:x_s+nx, y_s:y_s+ny] = prediction_cropped
            else:
                if x <= nx and y > ny:
                    slice_predictions[:,:, y_s:y_s+ny] = prediction_cropped[:,x_c:x_c+ x, :]
                elif x > nx and y <= ny:
                    slice_predictions[:,x_s:x_s + nx, :] = prediction_cropped[:,:, y_c:y_c + y]
                else:
                    slice_predictions[:,:, :] = prediction_cropped[:,x_c:x_c+ x, y_c:y_c + y]
            prediction = transform.resize(slice_predictions,
                                (4, mask.shape[0], mask.shape[1]),
                                order=1,
                                preserve_range=True,
                                anti_aliasing=True,
                                mode='constant')
            prediction = np.uint8(np.argmax(prediction, axis=0))
            #prediction = keep_largest_connected_components(prediction)
            predictions.append(prediction)
        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))
        dir_pred = os.path.join(output_folder, "predictions")
        makefolder(dir_pred)
        out_file_name = os.path.join(dir_pred, label_file)
        out_affine = mask_dat[1]
        out_header = mask_dat[2]

        save_nii(out_file_name, prediction_arr, out_affine, out_header)

        dir_gt = os.path.join(output_folder, "masks")
        makefolder(dir_gt)
        mask_file_name = os.path.join(dir_gt, label_file)
        save_nii(mask_file_name, mask_dat[0], out_affine, out_header)
    
    filenames_gt = sorted(glob.glob(os.path.join(dir_gt, '*')), key=natural_order)
    # filenames_gt = [f for f in filenames_gt if "01" in f]
    filenames_pred = sorted(glob.glob(os.path.join(dir_pred, '*')), key=natural_order)
    # filenames_pred = [f for f in filenames_pred if "01" in f]
    file_names = []
    structure_names = []
    # measures per structure:
    dices_list = []
    if dataset == 'MSCMR':
        structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}
        gt_value, pred_value = [500, 600, 200], [3, 1, 2]
    elif dataset == 'ACDC':
        structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}
        gt_value, pred_value = [3, 1, 2], [3, 1, 2]
    else:
        raise ValueError('Invalid dataset: {}'.format(dataset))        
    count = 0
    for p_gt, p_pred in zip(filenames_gt, filenames_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                            " {}, {}.".format(os.path.basename(p_gt),
                                            os.path.basename(p_pred)))
        # load ground truth and prediction
        gt, _, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        zooms = header.get_zooms()
        # calculate measures for each structure
        for value, struc in zip(gt_value, pred_value):
            gt_binary = (gt == value) * 1
            pred_binary = (pred == struc) * 1
            if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                dices_list.append(1)
            elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(gt_binary) > 0:
                dices_list.append(0)
                count += 1
            else:
                dices_list.append(dc(gt_binary, pred_binary))
            file_names.append(os.path.basename(p_pred))
            structure_names.append(structures_dict[struc])
    dices_np, l = np.asarray(dices_list), len(dices_list)
    print('Avg:{}+{}'.format(dices_np.mean(), dices_np.std()))
    print('LV:{}+{}'.format(dices_np[0:l:3].mean(), dices_np[0:l:3].std()))
    print('RV:{}+{}'.format(dices_np[1:l:3].mean(), dices_np[1:l:3].std()))
    print('Myo:{}+{}'.format(dices_np[2:l:3].mean(), dices_np[2:l:3].std()))    
    df = pd.DataFrame({'dice': dices_list, 'struc': structure_names, 'filename': file_names})
    csv_path = os.path.join(output_folder, "stats.csv")
    df.to_csv(csv_path)
    return df

