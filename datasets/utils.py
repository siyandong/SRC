from __future__ import division

import torch
import numpy as np
import random
import cv2
from imgaug import augmenters as iaa

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image

def get_depth(depth, calibration_extrinsics, intrinsics_color,
              intrinsics_depth_inv):
    """Return the calibrated depth image (7-Scenes). 
    Calibration parameters from DSAC (https://github.com/cvlab-dresden/DSAC) 
    are used.
    """
    img_height, img_width = depth.shape[0], depth.shape[1]
    depth_ = np.zeros_like(depth)
    x = np.linspace(0, img_width-1, img_width)
    y = np.linspace(0, img_height-1, img_height)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, (1, -1))
    yy = np.reshape(yy, (1, -1))
    ones = np.ones_like(xx)
    pcoord_depth = np.concatenate((xx, yy, ones), axis=0)
    depth = np.reshape(depth, (1, img_height*img_width))
    ccoord_depth = np.dot(intrinsics_depth_inv, pcoord_depth) * depth
    ccoord_depth[1,:] = - ccoord_depth[1,:]
    ccoord_depth[2,:] = - ccoord_depth[2,:]
    ccoord_depth = np.concatenate((ccoord_depth, ones), axis=0)
    ccoord_color = np.dot(calibration_extrinsics, ccoord_depth)
    ccoord_color = ccoord_color[0:3,:]
    ccoord_color[1,:] = - ccoord_color[1,:]
    ccoord_color[2,:] = depth

    pcoord_color = np.dot(intrinsics_color, ccoord_color)
    pcoord_color = pcoord_color[:,pcoord_color[2,:]!=0]
    pcoord_color[0,:] = pcoord_color[0,:]/pcoord_color[2,:]+0.5
    pcoord_color[0,:] = pcoord_color[0,:].astype(int)
    pcoord_color[1,:] = pcoord_color[1,:]/pcoord_color[2,:]+0.5
    pcoord_color[1,:] = pcoord_color[1,:].astype(int)
    pcoord_color = pcoord_color[:,pcoord_color[0,:]>=0]
    pcoord_color = pcoord_color[:,pcoord_color[1,:]>=0]
    pcoord_color = pcoord_color[:,pcoord_color[0,:]<img_width]
    pcoord_color = pcoord_color[:,pcoord_color[1,:]<img_height]

    depth_[pcoord_color[1,:].astype(int),
           pcoord_color[0,:].astype(int)] = pcoord_color[2,:]
    return depth_

def get_coord(depth, pose, intrinsics_color_inv):
    """Generate the ground truth scene coordinates from depth and pose.
    """
    img_height, img_width = depth.shape[0], depth.shape[1]
    mask = np.ones_like(depth)
    mask[depth==0] = 0
    mask = np.reshape(mask, (img_height, img_width,1))
    x = np.linspace(0, img_width-1, img_width)
    y = np.linspace(0, img_height-1, img_height)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, (1, -1))
    yy = np.reshape(yy, (1, -1))
    ones = np.ones_like(xx)
    pcoord = np.concatenate((xx, yy, ones), axis=0)
    depth = np.reshape(depth, (1, img_height*img_width))
    ccoord = np.dot(intrinsics_color_inv, pcoord) * depth
    ccoord = np.concatenate((ccoord, ones), axis=0)
    scoord = np.dot(pose, ccoord)
    scoord = np.swapaxes(scoord,0,1)
    scoord = scoord[:,0:3]
    scoord = np.reshape(scoord, (img_height, img_width,3))
    scoord = scoord * mask
    mask = np.reshape(mask, (img_height, img_width))
    return scoord, mask

def data_aug(img, mask, lbl, aug=True):
    img_h, img_w = img.shape[0:2]
    if aug:
        trans_x = random.uniform(-0.2,0.2)
        trans_y = random.uniform(-0.2,0.2)

        aug_add = iaa.Add(random.randint(-20,20))

        # default.
        scale=random.uniform(0.7,1.5)
        rotate=random.uniform(-30,30)
        shear=random.uniform(-10,10)

        aug_affine = iaa.Affine(scale=scale,rotate=rotate,shear=shear,
            translate_percent={"x": trans_x, "y": trans_y}) 
        aug_affine_lbl = iaa.Affine(scale=scale,rotate=rotate,shear=shear,
            translate_percent={"x": trans_x, "y": trans_y},order=0,cval=1) 
        img = aug_add.augment_image(img) 
    else:
        trans_x = random.randint(-3,4)
        trans_y = random.randint(-3,4)
        aug_affine = iaa.Affine(translate_px={"x": trans_x, "y": trans_y}) 
        aug_affine_lbl = iaa.Affine(translate_px={"x": trans_x, "y": trans_y},order=0,cval=1) 
                
    padding = torch.randint(0,255,size=(img_h,
                            img_w)).data.numpy().astype(np.uint8)
    padding_mask = np.ones((img_h,img_w)).astype(np.uint8)  
       
    img = aug_affine.augment_image(img)
    mask = aug_affine.augment_image(mask)
    mask = np.round(mask)
    lbl = aug_affine_lbl.augment_image(lbl)
    padding_mask = aug_affine.augment_image(padding_mask)
    img = img + (1-padding_mask) * padding

    return img, mask, lbl

def one_hot(x, N):  
    one_hot = torch.FloatTensor(N, x.size(0), x.size(1)).zero_()
    one_hot = one_hot.scatter_(0, x.unsqueeze(0), 1)           
    return one_hot

def to_tensor(img, mask, lbl_1, lbl_2, N1, N2):

    img = img / 255.
    #img = img * 2. - 1.

    img = torch.from_numpy(img).float()
    mask = torch.from_numpy(mask).float()
      
    lbl_1 = torch.from_numpy(lbl_1).long()
    lbl_2 = torch.from_numpy(lbl_2).long()
    
    lbl_1_oh = one_hot(lbl_1, N=N1)
    lbl_2_oh = one_hot(lbl_2, N=N2)

    return img, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh

def to_tensor_query(img, pose, lbls_query, mask):
    #img = img.transpose(2, 0, 1)    
    img = img / 255.
    #img = img * 2. - 1.
    img = torch.from_numpy(img).float()
    pose = torch.from_numpy(pose).float()
    lbls_query = torch.from_numpy(lbls_query).float()
    mask = torch.from_numpy(mask).float()
    return img, pose, lbls_query, mask