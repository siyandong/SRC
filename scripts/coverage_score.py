import numpy as np 
import open3d as o3d 
import os, cv2, copy, random
import sys
sys.path.append('..')
from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from utils import RGBD2PointCloud


random.seed(1)
num_low_shot = 20
load_test_pts = True       # False as default. True/False.
load_cov_vecs = True       # False as default. True/False.
use_refined_pose = False    # False as defualt. True/False.
voxel_size_reg = 0.02       # fixed.
voxel_size_cls = 0.10       # fixed.
voxel_size_qry = voxel_size_cls/4 # - - -
thresh_hit = voxel_size_cls*3 # fixed.
frame_folder_root = './datasets/7Scenes' # '/local/home/sidong/Desktop/dataset/7scenes/dsac'
pts_test_mask = '_05-16_xyzrgb_dcalib525_vs{}'.format(voxel_size_reg) + '_{}-test.txt'
cov_info_mask = '_06-01_covinfo_q-vs{:.4f}_hit-thresh{:.2f}'.format(voxel_size_qry, thresh_hit) + '_{}-train.txt'
# output_foler = './temp/_few-shot'
scenes = [
'chess',
'fire',
'heads',
'office',
'pumpkin',
'redkitchen',
'stairs',
]


# fixed params.
seqs_train = dict()
seqs_train['chess'] = [1,2,4,6]
seqs_train['fire'] = [1,2]
seqs_train['heads'] = [2]
seqs_train['office'] = [1,3,4,5,8,10]
seqs_train['pumpkin'] = [2,3,6,8]
seqs_train['redkitchen'] = [1,2,5,7,8,11,13]
seqs_train['stairs'] = [2,3,5,6]
seqs_test = dict()
seqs_test['chess'] = [3,5]
seqs_test['fire'] = [3,4]
seqs_test['heads'] = [1]
seqs_test['office'] = [2,6,7,9]
seqs_test['pumpkin'] = [1,7]
seqs_test['redkitchen'] = [3,4,6,12,14]
seqs_test['stairs'] = [1,4]
num_frames = dict()
num_frames['chess'] = 1000
num_frames['fire'] = 1000
num_frames['heads'] = 1000
num_frames['office'] = 1000
num_frames['pumpkin'] = 1000
num_frames['redkitchen'] = 1000
num_frames['stairs'] = 500
depth_mask = 'frame-{:06d}.depth_cali.png'
color_mask = 'frame-{:06d}.color.png'
pose_mask = 'frame-{:06d}.pose.txt'
refined_pose_mask = 'frame-{:06d}.pose_refi.txt'
image_width, image_height = 640, 480
intrinsics = np.array([[525, 0, 320],[0, 525, 240],[0, 0, 1]], dtype=float) # updated intrinsics.
def load_color_7scenes(color_path):
    color = cv2.imread(color_path, -1)
    color = color / 255.
    color = color[:,:,[2,1,0]]
    return color
def load_depth_7scenes(depth_path):
    depth = cv2.imread(depth_path, -1)
    depth[depth==65535] = 0
    depth = depth / 1000.
    return depth


if __name__ == '__main__':

    rgbd_pc = RGBD2PointCloud(image_width, image_height, intrinsics)
    ls_path = 'train_siyan_{}f_0516-refine-shuffle.txt'.format(num_low_shot)
    file = open(ls_path, 'w')
    file.close()

    fid_skip = 200
    fs_path = 'train_siyan_skip{}.txt'.format(fid_skip)
    file = open(fs_path, 'w')
    file.close()

    #################
    # for each scene.
    #################
    for scene in scenes:
        print('scene {}'.format(scene))


        ##################################
        # build pointcloud from test seqs.
        ##################################
        print('build pointcloud from test seqs')
        pts_path = '{}/{}'.format(frame_folder_root, pts_test_mask).format(scene)
        if load_test_pts: 
            scene_xyz = np.loadtxt(pts_path)[:,0:3]
        else:
            scene_pcd = None
            print('#seq {}'.format(len(seqs_test[scene])))
            for seq in seqs_test[scene]:
                for fid in tqdm(range(num_frames[scene])):
                    depth_path = '{}/{}/seq-{:02d}/{}'.format(frame_folder_root, scene, seq, depth_mask).format(fid)
                    depth = load_depth_7scenes(depth_path)
                    color_path = '{}/{}/seq-{:02d}/{}'.format(frame_folder_root, scene, seq, color_mask).format(fid)
                    color = load_color_7scenes(color_path)
                    pose_path = '{}/{}/seq-{:02d}/{}'.format(frame_folder_root, scene, seq, pose_mask).format(fid)
                    pose = np.loadtxt(pose_path)
                    xyzrgb = rgbd_pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=True)
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,0:3]))
                    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:6])
                    if scene_pcd: 
                        scene_pcd = scene_pcd + copy.deepcopy(pcd)
                    else: 
                        scene_pcd = copy.deepcopy(pcd)
                    scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size_reg)
            np.savetxt( pts_path, np.concatenate(( np.array(scene_pcd.points), np.array(scene_pcd.colors) ), axis=1) ) # save to file.
            scene_xyz = np.array(scene_pcd.points) 


        #########################################################
        # initialize coverage vector for each raw training frame.
        #########################################################
        print('initialize coverage vector for each raw training frame')
        scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_xyz[:,0:3]))
        scene_pcd_sample = scene_pcd.voxel_down_sample(voxel_size=voxel_size_cls)
        scene_xyz_sample = np.array(scene_pcd_sample.points)
        info_path = '{}/{}'.format(frame_folder_root, cov_info_mask).format(scene)
        if load_cov_vecs:
            cov_vecs = np.loadtxt(info_path)
        else:
            cov_vecs = []
            for seq in tqdm(seqs_train[scene]):
                frame_folder = '{}/{}/seq-{:02d}'.format(frame_folder_root, scene, seq)
                pose_folder = '{}/{}/seq-{:02d}'.format(frame_folder_root, scene, seq)
                for fid in tqdm(range(num_frames[scene])):
                    depth_path = '{}/{}'.format(frame_folder, depth_mask).format(fid)
                    depth = load_depth_7scenes(depth_path)
                    color_path = '{}/{}'.format(frame_folder, color_mask).format(fid)
                    color = load_color_7scenes(color_path)
                    if use_refined_pose: pose_path = '{}/{}'.format(pose_folder, refined_pose_mask).format(fid)
                    else: pose_path = '{}/{}'.format(pose_folder, pose_mask).format(fid)
                    pose = np.loadtxt(pose_path)
                    xyzrgb = rgbd_pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=True)
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,0:3]))
                    #pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:6])
                    pcd = pcd.voxel_down_sample(voxel_size=voxel_size_cls/2)
                    xyz = np.array(pcd.points)
                    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(scene_xyz_sample)
                    distances, indices = nbrs.kneighbors(xyz)
                    covered_ids = indices[distances<thresh_hit]
                    cov_vec = np.zeros((scene_xyz_sample.shape[0], 1))
                    cov_vec[covered_ids] = 1.0
                    cov_vecs.append(cov_vec)
            cov_vecs = np.array(cov_vecs).squeeze()
            np.savetxt(info_path, cov_vecs) # save to file.
        #'''


        #####################
        # coverage score.....
        #####################
        skip = 100
        fs_frames = []
        fs_cov = np.zeros(scene_xyz_sample.shape[0])
        for idx in range(0, len(seqs_train[scene]*num_frames[scene]), skip):
            fs_frames.append(int(idx))
            fs_cov = fs_cov + cov_vecs[idx]
            fs_cov[fs_cov>1.] = 1.
            cov_vecs = cov_vecs - cov_vecs[idx]
            cov_vecs[cov_vecs<0.] = 0.
            #print('coverage {:.2f}%'.format( fs_cov.sum() / fs_cov.shape[0] * 100))
        print('coverage {:.2f}%'.format( fs_cov.sum() / fs_cov.shape[0] * 100))
        print('max gain {:.2f}%'.format( cov_vecs.sum(axis=1).max() / fs_cov.shape[0] * 100 ))
        # while True:
        #     #print('coverage {:.2f}%'.format( fs_cov.sum() / fs_cov.shape[0] * 100))
        #     print('max gain {:.2f}%'.format( cov_vecs.sum(axis=1).max() / fs_cov.shape[0] * 100 ))
        #     input('???')    
        #     scores = cov_vecs.sum(axis=1)
        #     indices = np.argwhere(scores>=scores.max())
        #     idx = random.choice(indices)[0]
        #     #print('idx {}'.format(idx))
        #     fs_frames.append(int(idx))
        #     #if len(fs_frames)>=num_low_shot: break # fix frame number.
        #     fs_cov = fs_cov + cov_vecs[idx]
        #     fs_cov[fs_cov>1.] = 1.
        #     cov_vecs = cov_vecs - cov_vecs[idx]
        #     cov_vecs[cov_vecs<0.] = 0.
        # print('selected {} frames'.format(len(fs_frames)))
        # file = open(ls_path, 'a')
        # for count in range(len(fs_frames)):
        #     idx = fs_frames[count]
        #     seq = seqs_train[scene][int( idx / num_frames[scene] )]
        #     fid = idx % num_frames[scene]
        #     file.write('{} seq-{:02d} frame-{:06d}\n'.format(scene, seq, fid))
        # file.close()
        # #'''
        

        '''###################
        # uniform sampling.
        ###################
        idx = 0
        fs_frames = []
        while True:
            if idx >= len(seqs_train[scene]) * num_frames[scene]: break
            fs_frames.append(idx)
            idx+=fid_skip
        print('selected {} frames'.format(len(fs_frames)))
        fs_path = 'train_siyan_skip{}.txt'.format(fid_skip)
        file = open(fs_path, 'a')
        for count in range(len(fs_frames)):
            idx = fs_frames[count]
            seq = seqs_train[scene][int( idx / num_frames[scene] )]
            fid = idx % num_frames[scene]
            file.write('{} seq-{:02d} frame-{:06d}\n'.format(scene, seq, fid))
        file.close()
        #'''




        # #############################
        # # build the few shot dataset.
        # #############################
        # print('build the few shot dataset...')
        # fs_folder = '{}/{}/{}'.format(frame_folder_root, scene, output_foler)
        # if not os.path.exists(fs_folder): os.makedirs(fs_folder)
        # for count in range(len(fs_frames)):
        #     idx = fs_frames[count]
        #     seq = seqs_train[scene][int( idx / num_frames[scene] )]
        #     fid = idx % num_frames[scene]
        #     frame_folder = '{}/{}/seq-{:02d}'.format(frame_folder_root, scene, seq)
        #     if not use_refined_pose: pose_folder = '{}/{}/seq-{:02d}'.format(frame_folder_root, scene, seq)
        #     else: pose_folder = '{}/p-refine-vs*8*4*2_{}/seq-{:02d}'.format(pose_root, scene, seq)
        #     depth_path = '{}/{}'.format(frame_folder, depth_mask).format(fid)
        #     color_path = '{}/{}'.format(frame_folder, color_mask).format(fid)
        #     pose_path = '{}/{}'.format(pose_folder, pose_mask).format(fid)
        #     os.system('cp {} {}/{}'.format(depth_path, fs_folder, depth_mask).format(count))
        #     os.system('cp {} {}/{}'.format(color_path, fs_folder, color_mask).format(count))
        #     os.system('cp {} {}/{}'.format(pose_path, fs_folder, pose_mask).format(count))


        # ############################
        # # fusion of few shot frames.
        # ############################
        # print('fusion of few shot frames...')
        # fs_pts = None
        # for fid in range(len(fs_frames)):
        #     folder = '{}/{}/{}'.format(frame_folder_root, scene, output_foler)
        #     depth_path = '{}/{}'.format(folder, depth_mask).format(fid)
        #     depth = load_depth_7scenes(depth_path)
        #     color_path = '{}/{}'.format(folder, color_mask).format(fid)
        #     color = load_color_7scenes(color_path)
        #     pose_path = '{}/{}'.format(folder, pose_mask).format(fid)
        #     pose = np.loadtxt(pose_path)
        #     xyzrgb = rgbd_pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=True)
        #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,0:3]))
        #     pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:6])
        #     if fs_pts: fs_pts = fs_pts + copy.deepcopy(pcd)
        #     else: fs_pts = copy.deepcopy(pcd)
        # print('selected {} frames.'.format(len(fs_frames)))
        # np.savetxt('_fs_{}.txt'.format(scene), np.concatenate((np.array(fs_pts.points), np.array(fs_pts.colors)), axis=1) )
        #'''

