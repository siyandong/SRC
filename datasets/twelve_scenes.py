from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data

from .utils import *


class TwelveScenes(data.Dataset):
    def __init__(self, n_class, root, info, scene='apt2/bed', aug='True'):
        self.intrinsics_color = np.array([[572.0, 0.0, 320.0], [0.0, 572.0, 240.0], [0.0, 0.0, 1.0]])
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        self.n_class = n_class
        self.root = os.path.join(root, '12Scenes')
        self.info = info
        self.scene = scene
        self.aug = aug 
        self.obj_suffixes = ['.color.jpg', '.pose.txt', '.depth.png', '.label_n{}.png'.format(self.n_class**2)]
        self.obj_keys = ['color', 'pose', 'depth', 'label']
        
        file_path = os.path.join(self.root, self.info)
        with open(file_path, 'r') as f:
            self.frames = f.readlines()
            self.frames = [frame for frame in self.frames if self.scene in frame]


    def __len__(self):
        return len(self.frames)


    def __getitem__(self, index):
        _, frame = self.frames[index].rstrip('\n').split(' ')
        
        obj_files = ['{}{}'.format(frame, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, self.scene, 'data', 
                    obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
        
        img = read_image(objs['color'], grayscale=True)
        img = cv2.resize(img, (640, 480))
        
        lbl = cv2.imread(objs['label'],-1)
        depth = cv2.imread(objs['depth'],-1)   
        mask = np.ones_like(depth)
        mask[depth==0] = 0

        img, mask, lbl = data_aug(img, mask, lbl, self.aug)
               
        mask = mask[4::8,4::8].astype(np.float16)
        lbl = lbl[4::8,4::8].astype(np.int)
        img = img[None]
       
        lbl_1 = lbl // self.n_class
        lbl_2 = lbl % self.n_class 
        
        N1=self.n_class
        
        img, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh = to_tensor(img, mask, lbl_1, lbl_2, N1=N1, N2=N1)

        return img, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh  

