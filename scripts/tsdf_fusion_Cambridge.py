import sys
sys.path.append('..')
import open3d as o3d 
import numpy as np
import cv2, glob, copy
from tqdm import tqdm
from utils import RGBD2PointCloud


data_folder_root = './datasets/Cambridge'
train_file_path = './datasets/Cambridge/train_Cambridge_skip100.txt'
fid_skip = 1 # 1 or 100 default.
n_cluster = 64
voxel_size = 0.5
scenes = [
'GreatCourt',
'KingsCollege',
'OldHospital',
'ShopFacade',
'StMarysChurch',
]


depth_file_suffixes = dict()
depth_file_suffixes['GreatCourt'] = '.depth.tiff'
depth_file_suffixes['KingsCollege'] = '.depth.tiff'
depth_file_suffixes['OldHospital'] = '.depth.tiff'
depth_file_suffixes['ShopFacade'] = '.depth.png'
depth_file_suffixes['StMarysChurch'] = '.depth.tiff'
color_file_suffix = '.color.png'
pose_file_suffix = '.pose.txt'
label_file_suffix = '.label_siyan_{}.png'.format(n_cluster*n_cluster)
image_width, image_height = 852, 480
intrinsics = np.array([[744.375, 0.0,     426.0],
                       [0.0,     744.375, 240.0],
                       [0.0,     0.0,      1.0]])
def load_color_Cambridge(color_path):
    color = cv2.imread(color_path, -1)
    color = color / 255.
    color = color[:,:,[2,1,0]]
    color = cv2.resize(color, (image_width, image_height)) 
    return color
def load_depth_Cambridge(depth_path):
    depth = cv2.imread(depth_path, -1)
    depth = depth / 1000.
    return depth


if __name__ == '__main__':
    rgbd_pc = RGBD2PointCloud(image_width, image_height, intrinsics)


    #################
    # for each scene.
    #################
    for scene in scenes:
        print('scene {}'.format(scene))
        # the global volume.
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


        #######################
        # fuse few-shot frames.
        #######################
        frames_fs = None 
        with open(train_file_path, 'r') as f:
            frames_fs = f.readlines()
            frames_fs = [frame for frame in frames_fs if scene in frame]
        scene_pcd = None
        for fid in tqdm(range(len(frames_fs))):
            _, frame = frames_fs[fid].rstrip('\n').split(' ')
            depth_path = '{}/{}/train/{}{}'.format(data_folder_root, scene, frame, depth_file_suffixes[scene])
            color_path = depth_path.replace(depth_file_suffixes[scene], color_file_suffix)
            label_path = depth_path.replace(depth_file_suffixes[scene], label_file_suffix)
            pose_path = depth_path.replace(depth_file_suffixes[scene], pose_file_suffix)
            #print(depth_path, color_path, label_path, pose_path)
            pose = np.loadtxt(pose_path)
            depth = load_depth_Cambridge(depth_path)
            color = load_color_Cambridge(color_path)                                           # use original rgb color.
            label = cv2.imread(label_path, -1)
            #color = colors[label.reshape(-1)].reshape(label.shape[0], label.shape[1], 3) * 255  # use cluster color.
            # # debug:
            # cv2.imwrite('_temp.png', color.astype(np.uint8))
            # input('???')
            xyzrgb = rgbd_pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=True)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,0:3]))
            pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:6])
            if scene_pcd: 
                scene_pcd = scene_pcd + copy.deepcopy(pcd)
            else: 
                scene_pcd = copy.deepcopy(pcd)
            scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size)
        np.savetxt('xyzrgb_{}_fs.txt'.format(scene), np.concatenate((np.array(scene_pcd.points), np.array(scene_pcd.colors)), axis=1))


        '''#######################
        # fuse training frames.
        #######################
        scene_pcd = None
        folder = '{}/{}/train'.format(data_folder_root, scene)
        depth_file_list = sorted(glob.glob('{}/*{}'.format(folder, depth_file_suffixes[scene])))
        colors = np.random.rand(n_cluster*n_cluster, 3)
        for fid in tqdm(range(len(depth_file_list))):
            if not fid % fid_skip == 0: continue
            depth_path = depth_file_list[fid]
            color_path = depth_path.replace(depth_file_suffixes[scene], color_file_suffix)
            label_path = depth_path.replace(depth_file_suffixes[scene], label_file_suffix)
            pose_path = depth_path.replace(depth_file_suffixes[scene], pose_file_suffix)
            pose = np.loadtxt(pose_path)
            depth = load_depth_Cambridge(depth_path)
            color = load_color_Cambridge(color_path)                                           # use original rgb color.
            label = cv2.imread(label_path, -1)
            #color = colors[label.reshape(-1)].reshape(label.shape[0], label.shape[1], 3) * 255  # use cluster color.
            # # debug:
            # cv2.imwrite('_temp.png', color.astype(np.uint8))
            # input('???')
            xyzrgb = rgbd_pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=True)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,0:3]))
            pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:6])
            if scene_pcd: 
                scene_pcd = scene_pcd + copy.deepcopy(pcd)
            else: 
                scene_pcd = copy.deepcopy(pcd)
            scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size)
        np.savetxt('xyzrgb_{}.txt'.format(scene), np.concatenate((np.array(scene_pcd.points), np.array(scene_pcd.colors)), axis=1))
        #'''











            # color = o3d.io.read_image(color_path)
            # depth = o3d.io.read_image(depth_path)
            # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            #     color, depth, depth_trunc=100.0, convert_rgb_to_intensity=False)
            # volume.integrate(
            #     rgbd,
            #     o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240), # updated.
            #     np.linalg.inv(pose))




        '''
        #print(depth_file_list)
        # for seq in seqs_test[scene]:
        #     # for each frame.
        #     for fid in tqdm(range(num_frames[scene])):
        #         depth_path = '{}/{}/seq-{:02d}/{}'.format(data_folder_root, scene, seq, depth_mask).format(fid)
        #         color_path = '{}/{}/seq-{:02d}/{}'.format(data_folder_root, scene, seq, color_mask).format(fid)
        #         color = o3d.io.read_image(color_path)
        #         depth = o3d.io.read_image(depth_path)
        #         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #             color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        #         #pose_path = '{}/_tsdf-p-ref-vs*8*4*2_{}/seq-{:02d}/{}'.format(pose_root, scene, seq, pose_mask).format(fid) # debug.
        #         pose_path = '{}/{}/seq-{:02d}/{}'.format(data_folder_root, scene, seq, pose_mask).format(fid)
        #         pose = np.loadtxt(pose_path)
        #         volume.integrate(
        #             rgbd,
        #             o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240), # updated.
        #             np.linalg.inv(pose))
        #'''



        '''###########################
        # save mesh and pointcloud.
        ###########################
        mesh = volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh("tsdf-mesh_{}.ply".format(scene), mesh)
        pcd = volume.extract_point_cloud()
        xyz = np.array(pcd.points)
        rgb = np.array(pcd.colors)
        xyzrgb = np.concatenate((xyz, rgb), axis=1)
        np.savetxt('tsdf-xyzrgb_{}.txt'.format(scene), xyzrgb)
        # o3d.visualization.draw_geometries([pcd])
        # # mesh = volume.extract_triangle_mesh()
        # # mesh.compute_vertex_normals()
        # # o3d.visualization.draw_geometries([mesh])
        #'''



