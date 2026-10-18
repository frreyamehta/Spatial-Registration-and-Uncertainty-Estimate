import os
import cv2
import math
import numpy as np
from enum import Enum, auto, unique

import torch
from torch.utils.data import Dataset

#looks for 4 .npy image files
# paths[0] = target MOD
# paths[1] = target Landsat
# paths[2] = ref MOD
# paths[3] = ref Landsat
def get_pair_path(root_dir, target_dir_name, ref_dir_name):
    paths = [
        os.path.join(root_dir, 'MODIS',   f'MOD_{target_dir_name}.npy'),  # target MODIS
        os.path.join(root_dir, 'Landsat', f'LAN_{target_dir_name}.npy'),  # target Landsat
        os.path.join(root_dir, 'MODIS',   f'MOD_{ref_dir_name}.npy'),     # ref MODIS
        os.path.join(root_dir, 'Landsat', f'LAN_{ref_dir_name}.npy'),     # ref Landsat
    ]
    return paths

#loads the images
def load_image_pair(root_dir, target_dir_name, ref_dir_name):
    paths = get_pair_path(root_dir, target_dir_name, ref_dir_name)
    images = []
    for p in paths:
        im = np.load(p)
        images.append(im)

    return images


def transform_image(image, flip_num, rotate_num0, rotate_num):
    #valid pixel mask invalid -> 0 and valid -> 1
    # pixel vals = surface reflectance scaled by 10000 
    image_mask = np.ones(image.shape)
    negtive_mask = np.where(image < 0)
    inf_mask = np.where(image > 10000.)
#valid pixels contribute to loss 
    image_mask[negtive_mask] = 0.0
    image_mask[inf_mask] = 0.0
    image[negtive_mask] = 0.0
    image[inf_mask] = 10000.0
    image = image.astype(np.float32)
#horizontal flip
    if flip_num == 1:
        image = image[:, :, ::-1]
#rotation augmentation - inc training data diversity
    C, H, W = image.shape
    if rotate_num0 == 1:
        # -90
        if rotate_num == 2:
            image = image.transpose(0, 2, 1)[::-1, :]
        # 90
        elif rotate_num == 1:
            image = image.transpose(0, 2, 1)[:, ::-1]
        # 180
        else:
            image = image.reshape(C, H * W)[:, ::-1].reshape(C, H, W)
#convert to torch now its a pytorch tensor
    image = torch.from_numpy(image.copy())
    image_mask = torch.from_numpy(image_mask)
#normalized
    image.mul_(0.0001)
    image = image * 2 - 1
    return image, image_mask

#pytorch dataset
class PatchSet(Dataset):
    def __init__(self, root_dir, image_dates, image_size, patch_size):
        super(PatchSet, self).__init__()
        self.root_dir = root_dir
        self.image_dates = image_dates
        self.image_size = image_size
        self.patch_size = patch_size
#stride = 50% overlap b/w patches - good 4 spatial models
        PATCH_STRIDE = self.patch_size // 2
        end_h = (self.image_size[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
        end_w = (self.image_size[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
        h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
        w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
        if (self.image_size[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
            h_index_list.append(self.image_size[0] - self.patch_size)
        if (self.image_size[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
            w_index_list.append(self.image_size[1] - self.patch_size)
#tot training samples
        self.total_index = len(self.image_dates) * len(h_index_list) * len(w_index_list)

    def __getitem__(self, item):
        images = []
#load 1 .npy
        im = np.load(os.path.join(self.root_dir, str(item) + '.npy'))
    # 4 imgs x 6 bands = 24 channels 
    #split into [0:6]   → MOD_t
# [6:12]  → Landsat_t
# [12:18] → MOD_r
# [18:24] → Landsat_r
        for i in range(4):
            images.append(im[i * 6: i * 6 + 6, :, :])
        patches = [None] * len(images)
        masks = [None] * len(images)

        flip_num = np.random.choice(2)
        rotate_num0 = np.random.choice(2)
        rotate_num = np.random.choice(3)
        for i in range(len(patches)):
            im = images[i]
            im, im_mask = transform_image(im, flip_num, rotate_num0, rotate_num)
            patches[i] = im
            masks[i] = im_mask
# pixel valid only if it's valid in all images
        gt_mask = masks[0] * masks[1] * masks[2] * masks[3]

        return patches[0], patches[1], patches[2], patches[3], gt_mask

    def __len__(self):
        return self.total_index
