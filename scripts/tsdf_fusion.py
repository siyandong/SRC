import sys
sys.path.append('..')
import open3d as o3d 
import numpy as np
import cv2, copy
from tqdm import tqdm
from utils import RGBD2PointCloud


data_folder_root = './datasets/7Scenes'
tf_skip = 200
train_file_path = './datasets/7Scenes/train_siyan_skip{}.txt'.format(tf_skip) # './datasets/7Scenes/train_siyan_skip200+.txt' # './datasets/7Scenes/train_siyan_skip200.txt' # './datasets/7Scenes/train_siyan_05-10.txt'
n_cluster = 64
scenes = [
#'chess',
'fire',
#'heads',
'office',
'pumpkin',
#'redkitchen',
#'stairs',
]


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
depth_file_suffix = '.depth_cali.png'
color_file_suffix = '.color.png'
pose_file_suffix = '.pose.txt'
label_file_suffix = '.label_siyan_{}.png'.format(n_cluster*n_cluster)
image_width, image_height = 640, 480
intrinsics = np.array([[525, 0, 320],[0, 525, 240],[0, 0, 1]], dtype=float) # updated.
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
def load_global_coord(convertor, depth_path, pose_path, color_path=None):
    depth = load_depth_7scenes(depth_path)
    pose = np.loadtxt(pose_path)
    color = np.zeros((depth.shape[0], depth.shape[1], 3))
    if color_path: color = load_color_7scenes(color_path)
    coord = convertor.RGBD_pose_2pc(color, depth, pose, filter_invalid=False).reshape(depth.shape[0], depth.shape[1], 6)[:,:,0:3]
    return coord


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
            sdf_trunc=0.04, # default 0.04.
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        

        '''#################
        # fuse test seqs.
        #################
        print('#seq={}'.format(len(seqs_test[scene])))
        for seq in seqs_test[scene]:
            # for each frame.
            for fid in tqdm(range(num_frames[scene])):
                depth_path = '{}/{}/seq-{:02d}/{}'.format(data_folder_root, scene, seq, depth_mask).format(fid)
                color_path = '{}/{}/seq-{:02d}/{}'.format(data_folder_root, scene, seq, color_mask).format(fid)
                color = o3d.io.read_image(color_path)
                depth = o3d.io.read_image(depth_path)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
                #pose_path = '{}/_tsdf-p-ref-vs*8*4*2_{}/seq-{:02d}/{}'.format(pose_root, scene, seq, pose_mask).format(fid) # debug.
                pose_path = '{}/{}/seq-{:02d}/{}'.format(data_folder_root, scene, seq, pose_mask).format(fid)
                pose = np.loadtxt(pose_path)
                volume.integrate(
                    rgbd,
                    o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240), # updated.
                    np.linalg.inv(pose))
        #'''


        '''##################
        # fuse prediction.
        ##################
        fid_skip = 10
        prediction_folder = './temp/pred_{}'.format(scene)
        print('#seq={}'.format(len(seqs_test[scene])))
        for s_idx in range(len(seqs_test[scene])):
            seq = seqs_test[scene][s_idx]
            # for each frame.
            for fid in tqdm(range(0, num_frames[scene], fid_skip)):
                depth_path = '{}/{}/seq-{:02d}/{}'.format(data_folder_root, scene, seq, depth_mask).format(fid)
                color_path = '{}/{}/seq-{:02d}/{}'.format(data_folder_root, scene, seq, color_mask).format(fid)
                color = o3d.io.read_image(color_path)
                depth = o3d.io.read_image(depth_path)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
                pose_path = '{}/{}.pose-pred.txt'.format(prediction_folder, int(s_idx*num_frames[scene]/fid_skip + fid/fid_skip))
                pose = np.loadtxt(pose_path)
                volume.integrate(
                    rgbd,
                    o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240), # updated.
                    np.linalg.inv(pose))
        #'''


        '''##################################
        # color coding by few-shot frames. 
        ##################################
        frames_fs = None
        with open(train_file_path, 'r') as f:
            frames_fs = f.readlines()
            frames_fs = [frame for frame in frames_fs if scene in frame]
        # bbx and lab color.
        scene_pcd = None
        label2coords = dict()
        for idx in tqdm(range(len(frames_fs))):
            frame = frames_fs[idx].rstrip('\n')
            scene, seq_id, frame_id = frame.split(' ')
            depth_path = '{}/{}/{}/{}{}'.format(data_folder_root, scene, seq_id, frame_id, depth_file_suffix)
            color_path = depth_path.replace(depth_file_suffix, color_file_suffix)
            pose_path = depth_path.replace(depth_file_suffix, pose_file_suffix)
            depth = load_depth_7scenes(depth_path)
            color = load_color_7scenes(color_path)
            pose = np.loadtxt(pose_path)
            # coords.
            xyzrgb = rgbd_pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=True)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,0:3]))
            if scene_pcd is None: scene_pcd = copy.deepcopy(pcd)
            else: scene_pcd += pcd
            scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.02)
            # lab.
            label_path = '{}/{}/{}/{}{}'.format(data_folder_root, scene, seq_id, frame_id, label_file_suffix)
            label = cv2.imread(label_path, -1).reshape(-1).astype(np.int64)
            coord = rgbd_pc.RGBD_pose_2pc(color, depth, pose, filter_invalid=False)[:,0:3]
            for pid in range(label.shape[0]):
                #lab = label[pid]                # 4096. - - - - - - - - - -
                lab = label[pid] // n_cluster   # 64. - - - - - - - - - -
                if lab not in label2coords.keys(): label2coords[lab] = []
                label2coords[lab].append(coord[pid])
        scene_coords = np.array(scene_pcd.points)
        x_min, x_max = scene_coords[:,0].min(), scene_coords[:,0].max()
        y_min, y_max = scene_coords[:,1].min(), scene_coords[:,1].max()
        z_min, z_max = scene_coords[:,2].min(), scene_coords[:,2].max()
        #colors = np.zeros((n_cluster*n_cluster, 3)) # 4096. - - - - - - - - - -
        colors = np.zeros((n_cluster, 3))           # 64. - - - - - - - - - -
        for key in label2coords:
            xyz = np.array(label2coords[key])
            R = (xyz[:,0] - x_min) / (x_max-x_min)
            G = (xyz[:,1] - y_min) / (y_max-y_min)
            B = (xyz[:,2] - z_min) / (z_max-z_min)
            R, G, B = R.mean(), G.mean(), B.mean()
            if R < 0.: R = 0.
            if R > 1.: R = 1.
            if G < 0.: G = 0.
            if G > 1.: G = 1.
            if B < 0.: B = 0.
            if B > 1.: B = 1.
            colors[key] = np.array([R, G, B])
        # random color coding.
        #colors = np.random.rand(n_cluster*n_cluster, 3)
        #colors = np.random.rand(n_cluster, 3)
        #np.savetxt('_colors_coding_{}_l2.txt'.format(n_cluster), colors)
        np.savetxt('_colors_coding_{}_l1.txt'.format(n_cluster), colors)
        input('???')
        #'''


        colors = np.loadtxt('_colors_{}_l2.txt'.format(n_cluster)) # - - - - - - - - - -
        #colors = np.loadtxt('_colors_{}_l1.txt'.format(n_cluster)) # - - - - - - - - - -


        #######################
        # fuse few-shot frames.
        #######################
        frames_fs = None
        with open(train_file_path, 'r') as f:
            frames_fs = f.readlines()
            frames_fs = [frame for frame in frames_fs if scene in frame]
        for idx in tqdm(range(len(frames_fs))):
            frame = frames_fs[idx].rstrip('\n')
            scene, seq_id, frame_id = frame.split(' ')
            label_path = '{}/{}/{}/{}{}'.format(data_folder_root, scene, seq_id, frame_id, label_file_suffix)
            label = cv2.imread(label_path, -1)
            color = colors[label.reshape(-1)].reshape(label.shape[0], label.shape[1], 3) * 255                          # cluster colors 4096.
            #color = colors[(label.astype(np.int64)//64).reshape(-1)].reshape(label.shape[0], label.shape[1], 3) * 255   # cluster colors 64.
            cv2.imwrite('_temp.png', color.astype(int))
            depth_path = '{}/{}/{}/{}{}'.format(data_folder_root, scene, seq_id, frame_id, depth_file_suffix)
            color_path = '{}/{}/{}/{}{}'.format(data_folder_root, scene, seq_id, frame_id, color_file_suffix)   # color use raw rgb.
            #color_path = '_temp.png'                                                                            # color use cluster colors.
            color = o3d.io.read_image(color_path)
            depth = o3d.io.read_image(depth_path)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
            pose_path = '{}/{}/{}/{}{}'.format(data_folder_root, scene, seq_id, frame_id, pose_file_suffix)
            pose = np.loadtxt(pose_path)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    image_width, image_height, 
                    intrinsics[0,0], intrinsics[1,1], 
                    intrinsics[0,2], intrinsics[1,2]), 
                np.linalg.inv(pose))
        #'''




        ###########################
        # save mesh and pointcloud.
        ###########################
        mesh = volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh("tsdf-mesh_{}.ply".format(scene), mesh)
        pcd = volume.extract_point_cloud()
        xyz = np.array(pcd.points)
        rgb = np.array(pcd.colors)
        xyzrgb = np.concatenate((xyz, rgb), axis=1)
        np.savetxt('tsdf-xyzrgb_skip{}_{}.txt'.format(tf_skip, scene), xyzrgb)
        # o3d.visualization.draw_geometries([pcd])
        # # mesh = volume.extract_triangle_mesh()
        # # mesh.compute_vertex_normals()
        # # o3d.visualization.draw_geometries([mesh])



