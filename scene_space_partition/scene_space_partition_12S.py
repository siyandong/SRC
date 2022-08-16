import cv2, sys, copy, time, torch, argparse, pickle
import numpy as np 
import open3d as o3d
from tqdm import tqdm
from sklearn.cluster import KMeans
from .utils import *


class SceneSpacePartition_12S(object):
    def __init__(self, dataset_path, scene, train_file_path, n_class):
        super(SceneSpacePartition_12S, self).__init__()
        self.data_folder = '{}/12Scenes/{}'.format(dataset_path, scene)
        self.scene = scene
        self.frames_fs = None
        with open(train_file_path, 'r') as f:
            self.frames_fs = f.readlines()
            self.frames_fs = [frame for frame in self.frames_fs if scene in frame]
        self.n_class = n_class
        self.label_file_suffix = '.label_n{}.png'.format(n_class*n_class)
        self.final_centers = None

        # fixed params.
        self.depth_file_suffix = '.depth.png'
        self.color_file_suffix = '.color.jpg'
        self.pose_file_suffix = '.pose.txt'
        self.image_width, self.image_height = 640, 480
        self.intrinsics = np.array([[572.0, 0.0, 320.0], [0.0, 572.0, 240.0], [0.0, 0.0, 1.0]])
        self.rgbd2pc = RGBD2PointCloud(self.image_width, self.image_height, self.intrinsics)
        self.voxel_size = 0.01

    def load_color(self, color_path):
        color = cv2.imread(color_path, -1)
        color = color / 255.
        color = color[:,:,[2,1,0]]
        color = cv2.resize(color, (self.image_width, self.image_height))
        return color

    def load_depth(self, depth_path):
        depth = cv2.imread(depth_path, -1)
        depth[depth==65535] = 0
        depth = depth / 1000.
        return depth

    def load_global_coord(self, depth_path, pose_path, color_path=None):
        depth = self.load_depth(depth_path)
        pose = np.loadtxt(pose_path)
        color = np.zeros((depth.shape[0], depth.shape[1], 3))
        if color_path: color = self.load_color(color_path)
        coord = self.rgbd2pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=False).reshape(depth.shape[0], depth.shape[1], 6)[:,:,0:3]
        return coord


    def partition(self):
        
        ############################
        # build few-shot pointcloud.
        ############################
        print('build few-shot pointcloud, total {} frames'.format(len(self.frames_fs)))
        fs_pcd = None
        for idx in tqdm(range(len(self.frames_fs))):
            frame = self.frames_fs[idx].rstrip('\n')
            _, frame_id = frame.split(' ')
            depth_path = '{}/data/{}{}'.format(self.data_folder, frame_id, self.depth_file_suffix)
            color_path = '{}/data/{}{}'.format(self.data_folder, frame_id, self.color_file_suffix)
            pose_path =  '{}/data/{}{}'.format(self.data_folder, frame_id, self.pose_file_suffix)
            depth = self.load_depth(depth_path)
            color = self.load_color(color_path)
            pose = np.loadtxt(pose_path)
            xyzrgb = self.rgbd2pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=True)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,0:3]))
            pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:6])
            if fs_pcd: fs_pcd = fs_pcd + copy.deepcopy(pcd)
            else: fs_pcd = copy.deepcopy(pcd)
            if idx % 35 == 34: fs_pcd = fs_pcd.voxel_down_sample(voxel_size=self.voxel_size) # to deal with large training set.
        fs_pcd = fs_pcd.voxel_down_sample(voxel_size=self.voxel_size)

        ##########################
        # hierarchical clustering.
        ##########################
        print('hierarchical clustering')
        self.final_centers = []
        fs_coord = np.array(fs_pcd.points)[:,0:3]
        level_1 = KMeans(n_clusters=self.n_class, random_state=0).fit(fs_coord)
        for l1_cid in tqdm(range(self.n_class)):
            coord = fs_coord[level_1.labels_==l1_cid]
            level_2 = KMeans(n_clusters=self.n_class, random_state=0).fit(coord)
            for l2_cid in range(self.n_class):
                self.final_centers.append(level_2.cluster_centers_[l2_cid])
        self.final_centers = np.array(self.final_centers)
        #np.savetxt(, self.final_centers)
        return self.final_centers


    def label(self):
        cluster_centers = torch.Tensor(self.final_centers).cuda()
        
        ########################
        # label few-shot frames.
        ########################
        print('label the few-shot training set')
        for idx in tqdm(range(len(self.frames_fs))):
            frame = self.frames_fs[idx].rstrip('\n')
            _, frame_id = frame.split(' ')
            depth_path = '{}/data/{}{}'.format(self.data_folder, frame_id, self.depth_file_suffix)
            color_path = '{}/data/{}{}'.format(self.data_folder, frame_id, self.color_file_suffix)
            pose_path =  '{}/data/{}{}'.format(self.data_folder, frame_id, self.pose_file_suffix)
            depth = self.load_depth(depth_path)
            color = self.load_color(color_path)
            pose = np.loadtxt(pose_path)
            xyzrgb = self.rgbd2pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=False)
            # compute the closest cluster center for each pixel.
            xyz = torch.Tensor(xyzrgb[:,0:3]).cuda()
            dist_mat = torch.cdist(xyz[None,:,:], cluster_centers[None,:,:]).squeeze()
            label = dist_mat.argmin(dim=1).reshape(self.image_height, self.image_width).cpu().numpy().astype(np.uint16)
            label_path = '{}/data/{}{}'.format(self.data_folder, frame_id, self.label_file_suffix)
            cv2.imwrite(label_path, label)
            xyz, dist_mat, label = None, None, None


    def run(self):
        self.partition()
        self.label()

