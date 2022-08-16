from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data

from .utils import *


class Cambridge(data.Dataset):
    def __init__(self, n_class, root, info, dataset='Cambridge', scene='GreatCourt', split='train', aug='True'):
        self.intrinsics_color = np.array([[744.375, 0.0, 426.0], [0.0, 744.375, 240.0], [0.0, 0.0, 1.0]])
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)    
        self.n_class = n_class
        self.info = info
        self.dataset = dataset
        self.scene = scene
        self.root = os.path.join(root, 'Cambridge', self.scene)
        self.aug = aug
        self.split = split
        self.obj_suffixes = ['.color.png', '.pose.txt', '.depth.png', '.label_n{}.png'.format(self.n_class*self.n_class)]
        self.obj_keys = ['color', 'pose', 'depth', 'label']
        file_path = '{}/Cambridge/{}'.format(root, info)
        with open(file_path, 'r') as f: 
            self.frames = f.readlines()
            self.frames = [frame for frame in self.frames \
                if self.scene in frame]

    def __len__(self):
        return len(self.frames)


    def __getitem__(self, index):
        scene, frame = self.frames[index].rstrip('\n').split(' ') 

        obj_files = ['{}{}'.format(frame, obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, self.split, obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
             
        img = read_image(objs['color'], grayscale=True)
        img = cv2.resize(img, (852, 480)) 
        pose = np.loadtxt(objs['pose'])

        if self.split == 'test':

            img = img / 255.
            img = torch.from_numpy(img).float()
            pose = torch.from_numpy(pose).float()
            return img, pose

        lbl = cv2.imread(objs['label'],-1) 
        
        depth_fn = objs['depth']
        if self.scene != 'ShopFacade':
            depth_fn = depth_fn.replace('.png','.tiff')
        depth = cv2.imread(depth_fn,-1)      
            
        pose[0:3,3] = pose[0:3,3] * 1000
            
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)
        img, mask, lbl = data_aug(img, mask, lbl, self.aug)
        
        img_h, img_w = img.shape[0:2]
        th, tw = 480, 640
        x1 = random.randint(0, img_w - tw)
        y1 = random.randint(0, img_h - th)
        
        img = img[y1:y1+th,x1:x1+tw]
        img = img[None]
        mask = mask[y1:y1+th,x1:x1+tw]
        lbl = lbl[y1:y1+th,x1:x1+tw]
        
        mask = mask[4::8,4::8].astype(np.float16)
        lbl = lbl[4::8,4::8].astype(np.int)

        lbl_1 = lbl // self.n_class
        lbl_2 = lbl % self.n_class
        
        img, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh = to_tensor(img, mask, lbl_1, lbl_2, N1=self.n_class, N2=self.n_class)

        return img, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh


