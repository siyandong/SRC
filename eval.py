from __future__ import division

import os, sys, copy, argparse
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


def get_pose_err(pose_gt, pose_est):
    transl_err = np.linalg.norm(pose_gt[0:3,3]-pose_est[0:3,3])
    rot_err = pose_est[0:3,0:3].T.dot(pose_gt[0:3,0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return transl_err, rot_err[0]

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




def eval(args):
    scenes_7S = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen','stairs']
    scenes_Cambridge = ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']

    if args.dataset == '7S':
        if args.scene not in scenes_7S:
            print('selected scene is not valid.')
            sys.exit()

    if args.dataset == 'Cambridge':
        if args.scene not in scenes_Cambridge:
            print('selected scene is not valid.')
            sys.exit()

    # datasets.
    dataset = get_dataset(args.dataset)
    dataset = dataset(
        n_class = args.n_class,
        root=args.data_path, 
        info=args.test_info,
        dataset=args.dataset, 
        scene=args.scene, 
        split='test')
    intrinsics_color = dataset.intrinsics_color
    dataloader = data.DataLoader(dataset, 
        batch_size=1, 
        num_workers=4, 
        shuffle=False)

    # pose optimizer.
    pose_solver = pnpransac(intrinsics_color[0,0], intrinsics_color[1,1], intrinsics_color[0,2], intrinsics_color[1,2])

    # network.
    torch.set_grad_enabled(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    extractor = SuperPoint()
    extractor_path = os.path.join(args.checkpoint, 'extractor.pkl') 
    extractor_state = torch.load(extractor_path, map_location=device)['model_state']
    extractor.load_state_dict(extractor_state)
    extractor.to(device)
    extractor.eval()
    
    classifier = get_model(args.model, n_class=args.n_class)
    classifier_path = os.path.join(args.checkpoint, 'classifier.pkl')
    classifier_state = torch.load(classifier_path, map_location=device)['model_state']
    classifier.load_state_dict(classifier_state)
    classifier.to(device)
    classifier.eval()

    # leaf coordinates mapping.
    dataset_path = args.data_path
    if args.dataset == '7S':
        dataset_path+='/7Scenes'
    elif args.dataset == 'Cambridge':
        dataset_path+='/Cambridge'
    # load clustered coordiantes at each region
    leaf_coords_file_path = '{}/{}/_leaf_coords_n{}.npy'.format(dataset_path, args.scene, args.n_class*args.n_class)
    leaf_coords = np.load(leaf_coords_file_path)

    # start evaluation...
    rot_err_list = []
    transl_err_list = []
    accuracy_list = []
    x = np.linspace(4, 640-4, 80) + 106 * (args.dataset == 'Cambridge')
    y = np.linspace(4, 480-4, 60)
    xx, yy = np.meshgrid(x, y)
    pcoord = np.concatenate((np.expand_dims(xx,axis=2), 
            np.expand_dims(yy,axis=2)), axis=2)

    #for enm_idx, (img, pose, gt_lbls, gt_mask) in enumerate(tqdm(dataloader)):
    for enm_idx, (img, pose) in enumerate(tqdm(dataloader)):

        if args.dataset == 'Cambridge':
            img = img[:,:,106:106+640].to(device)
        else:
            img = img.to(device)

        img = torch.unsqueeze(img, 0)
        feature_map = extractor(img)
        lbl_2, lbl_1 = classifier(feature_map)
        lbl_1 = torch.argmax(lbl_1, dim=1)
        lbl_2 = torch.argmax(lbl_2, dim=1)
        lbl = (lbl_1 * args.n_class + lbl_2)


        lbl = lbl.cpu().data.numpy()[0,:,:]
        coords_all = leaf_coords[np.reshape(lbl,(-1)), :, :] 
 
        q_coords = 10
        if args.dataset == '7S':
            h_hypo = 256
        elif args.dataset == 'Cambridge':
            h_hypo = 512
        coords_all = np.ascontiguousarray(coords_all)
        coords_ransac = np.reshape(coords_all[:,1:,:3], (-1,q_coords,3)).astype(np.float64)
        pcoord = np.ascontiguousarray(pcoord)
        pcorrd_ransac = np.reshape(pcoord, (-1,2)).astype(np.float64)

        rot, transl = pose_solver.RANSAC_one2many(
            pcorrd_ransac, 
            coords_ransac,
            h_hypo
            )

        pose_gt = pose.data.numpy()[0,:,:]
        pose_est = np.eye(4)
        pose_est[0:3,0:3] = cv2.Rodrigues(rot)[0].T
        pose_est[0:3,3] = -np.dot(pose_est[0:3,0:3], transl)

        transl_err, rot_err = get_pose_err(pose_gt, pose_est)

        # debug pose.
        debug_folder = './temp/pred'
        if os.path.exists(debug_folder):
            np.savetxt('{}/{:06d}.pose-pred.txt'.format(debug_folder, enm_idx), pose_est)

        rot_err_list.append(rot_err)
        transl_err_list.append(transl_err)

        print('Pose error: {:.2f}cm, {:.2f}\u00b0'.format(transl_err*100, rot_err))

    accuracy_array = np.array(accuracy_list)
    results = np.array([transl_err_list, rot_err_list]).T
    np.savetxt(os.path.join(args.output, 'pose_err_{}_{}_{}.txt'.format(args.dataset, args.scene.replace('/','.'), args.model)), results)
    if args.dataset != 'Cambridge':
        print('Accuracy: {:.2f}%'.format(np.sum((results[:,0] <= 0.05) * (results[:,1] <= 5)) * 1. / len(results) * 100))
    print('Median pose error: {:.2f}cm, {:.2f}\u00b0'.format(np.median(results[:,0]*100), np.median(results[:,1])))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='infer camera pose by the SRC model + pnp with ransac.')
    parser.add_argument('--model', type=str, default='net1', choices=('net0', 'net1'), help='choose a network model.')
    parser.add_argument('--n_class', type=int, default=64, help='number of classes each level.')
    parser.add_argument('--data_path', type=str, required=True, help='path to the dataset.')
    parser.add_argument('--dataset', type=str, default='7S', choices=('7S', 'Cambridge'), help='choose a dataset.')
    parser.add_argument('--scene', type=str, default='chess', help='choose a scene from the dataset.')
    parser.add_argument('--test_info', type=str, default='test.txt', help='the file that contains the list of test images.')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to a checkpoint.')
    parser.add_argument('--output', type=str, default='./', help='output directory.')
    args = parser.parse_args()
    eval(args)

