import sys
sys.path.append('..')
import numpy as np 
import open3d as o3d 
import os, cv2, copy, random
from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from utils import RGBD2PointCloud


load_test_pts = False # False as default. True/False.
voxel_size_reg = 0.02
rad_for_norm_est = voxel_size_reg * 2
frame_folder_root = './datasets/7Scenes' # '/local/home/sidong/Desktop/dataset/7scenes/dsac'
pts_test_mask = '_05-16_xyzrgb_dcalib525_vs{}'.format(voxel_size_reg) + '_{}-test.txt'
scenes = [
#'chess',
#'fire',
'heads',
#'office',
#'pumpkin',
#'redkitchen',
#'stairs',
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
def ICP(source, target, thresh, mode='p2l', Rt_init=np.identity(4)):
    if mode == 'p2p':
        # point-to-point ICP.
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, thresh, Rt_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        transformation = reg_p2p.transformation
    elif mode == 'p2l':
        # point-to-plane ICP.
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, thresh, Rt_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transformation = reg_p2l.transformation
    elif mode =='color':
        # color ICP.
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source, target, thresh, Rt_init,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=24)
            )
        transformation = result_icp.transformation
    return transformation
def ICP_frame2model(crt_pcd, fusion, thresh, mode='p2l', Rt_init=np.identity(4)):
    source, target = copy.deepcopy(crt_pcd), copy.deepcopy(fusion)
    pose_abs = ICP(source, target, thresh, mode, Rt_init)
    return pose_abs


if __name__ == '__main__':

    rgbd_pc = RGBD2PointCloud(image_width, image_height, intrinsics)

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


        ##############################################
        # ICP pose refinement each raw training frame.
        ##############################################
        print('ICP pose refinement each raw training frame')
        scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_xyz[:,0:3]))
        for seq in tqdm(seqs_train[scene]):
            frame_folder = '{}/{}/seq-{:02d}'.format(frame_folder_root, scene, seq)
            pose_folder = '{}/{}/seq-{:02d}'.format(frame_folder_root, scene, seq)
            for fid in tqdm(range(num_frames[scene])):
                depth_path = '{}/{}'.format(frame_folder, depth_mask).format(fid)
                depth = load_depth_7scenes(depth_path)
                color_path = '{}/{}'.format(frame_folder, color_mask).format(fid)
                color = load_color_7scenes(color_path)
                pose_path = '{}/{}'.format(pose_folder, pose_mask).format(fid)
                pose = np.loadtxt(pose_path)
                xyzrgb = rgbd_pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=True)
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,0:3]))
                #pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:6])
                # refine pose.
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad_for_norm_est, max_nn=30))
                pcd.orient_normals_towards_camera_location(pose[0:3,3])
                pcd.normalize_normals()
                scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad_for_norm_est, max_nn=30))
                scene_pcd.orient_normals_towards_camera_location(pose[0:3,3])
                scene_pcd.normalize_normals()
                Rt = np.identity(4)
                Rt = ICP_frame2model(pcd, scene_pcd, voxel_size_reg*4, 'p2l', Rt)
                pose_refined = Rt @ pose
                refined_pose_path = '{}/{}'.format(pose_folder, refined_pose_mask).format(fid)
                np.savetxt(refined_pose_path, pose_refined)
        #'''


        '''#######################
        # fuse training frames.
        #######################
        print('fuse training frames')
        scene_pcd = None
        #for seq in tqdm(seqs_train[scene]):
        for seq in tqdm([5]): # debug.
            frame_folder = '{}/{}/seq-{:02d}'.format(frame_folder_root, scene, seq)
            pose_folder = '{}/{}/seq-{:02d}'.format(frame_folder_root, scene, seq)
            for fid in tqdm(range(num_frames[scene])):
                depth_path = '{}/{}'.format(frame_folder, depth_mask).format(fid)
                depth = load_depth_7scenes(depth_path)
                color_path = '{}/{}'.format(frame_folder, color_mask).format(fid)
                color = load_color_7scenes(color_path)
                pose_path = '{}/{}'.format(pose_folder, pose_mask).format(fid)  # raw pose.
                pose = np.loadtxt(pose_path)                                    # raw pose.
                # refined_pose_path = '{}/{}'.format(pose_folder, refined_pose_mask).format(fid)  # refined pose.
                # pose = np.loadtxt(refined_pose_path)                                            # refined pose.
                xyzrgb = rgbd_pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=True)
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,0:3]))
                pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:6])
                if scene_pcd is None: scene_pcd = copy.deepcopy(pcd)
                else: scene_pcd += pcd
                scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size_reg)
            np.savetxt('_temp_raw.txt', np.concatenate((np.array(scene_pcd.points), np.array(scene_pcd.colors)), axis=1))
            input('???')
        #'''



