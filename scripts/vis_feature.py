from __future__ import division

import sys
sys.path.append('..')
import os, copy
import argparse
import numpy as np
import cv2
import torch
from torch.utils import data

sys.path.insert(0, './pnpransac')
from pnpransac import pnpransac
from models import get_model
from datasets import get_dataset
from models.superpoint import SuperPoint
from tqdm import tqdm


def vis_feature(feature_map, shape=None):
    vis = copy.deepcopy(feature_map)
    for i in range(3): 
        vis[:,:,i] = (vis[:,:,i] - vis[:,:,i].min()) / (vis[:,:,i].max() - vis[:,:,i].min())
    vis = vis*255
    if shape:
        vis = cv2.resize(vis, shape, cv2.INTER_NEAREST)
    return vis
def vis_coord(coord, shape=None):
    vis = copy.deepcopy(coord)
    # normalize by self min and max.
    for i in range(3): 
        vis[:,:,i] = (vis[:,:,i] - coord[:,:,i].min()) / (coord[:,:,i].max() - coord[:,:,i].min())
    vis = vis*255
    if shape:
        vis = cv2.resize(vis, shape, cv2.INTER_NEAREST)
    return vis


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="?")
    args = parser.parse_args()
    args.n_class = 64
    args.data_path = './datasets'
    args.dataset = '7S'
    args.scene = 'chess'
    args.model = 'net0'
    args.aug = False
    args.batch_size = 1
    args.n_cluster = 64
    args.checkpoint = './checkpoints/7S-chess-net0-initlr0.0005-iters30000-bsize1-aug1-0'

    # colors_l1 = np.random.rand(args.n_cluster, 3)
    # colors_l2 = np.random.rand(args.n_cluster*args.n_cluster, 3)
    # np.savetxt('_colors_{}_l1.txt'.format(args.n_cluster), colors_l1)
    # np.savetxt('_colors_{}_l2.txt'.format(args.n_cluster), colors_l2)
    colors_l1 = np.loadtxt('_colors_{}_l1.txt'.format(args.n_cluster))
    colors_l2 = np.loadtxt('_colors_{}_l2.txt'.format(args.n_cluster))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, args.dataset, n_class=args.n_class)
    mapping_path = os.path.join(args.checkpoint, 'model_300.pkl')
    #extractor_path = os.path.join(args.checkpoint, 'extractor.pkl') 
    model_state = torch.load(mapping_path, map_location=device)['model_state']
    model.load_state_dict(model_state)
    model#.to(device)
    model.eval()
    extractor = SuperPoint()
    extractor.eval()

    dataset = get_dataset('7S')
    dataset = dataset(n_class=args.n_cluster, root=args.data_path, dataset=args.dataset, scene=args.scene, model=args.model, aug=args.aug)
    trainloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    
    for _, (img, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh) in enumerate(tqdm(trainloader)):
        feature_map = extractor(img)
        lbl_2_pred, lbl_1_pred = model(feature_map, lbl_1_oh, lbl_2_oh)
        lbl_1_pred = torch.argmax(lbl_1_pred, dim=1)
        lbl_2_pred = torch.argmax(lbl_2_pred, dim=1)


        # vis feature and pred.
        feature_map = feature_map[0,0:3,:,:].permute(1,2,0).detach().numpy()
        vis = vis_feature(feature_map)
        cv2.imwrite('feat.png', vis)
        color_l1 = (colors_l1[lbl_1_pred.squeeze().numpy().reshape(-1)].reshape(60,80,3) * 255).astype(np.int64)
        cv2.imwrite('_pred_l1.png', color_l1)
        color_l2 = (colors_l2[lbl_1_pred.squeeze().numpy().reshape(-1)*args.n_cluster + lbl_2_pred.squeeze().numpy().reshape(-1)].reshape(60,80,3) * 255).astype(np.int64)
        cv2.imwrite('_pred_l2.png', color_l2)
        

        # vis gt label.
        # print(lbl_1.shape)
        # print(lbl_2.shape)
        # print(lbl_1_oh.shape)
        # print(lbl_2_oh.shape)
        color_l1 = (colors_l1[lbl_1.squeeze().numpy().reshape(-1)].reshape(60,80,3) * 255).astype(np.int64)
        cv2.imwrite('_gt_l1.png', color_l1)
        color_l2 = (colors_l2[lbl_1.squeeze().numpy().reshape(-1)*args.n_cluster + lbl_2.squeeze().numpy().reshape(-1)].reshape(60,80,3) * 255).astype(np.int64)
        cv2.imwrite('_gt_l2.png', color_l2)


        input('???')











