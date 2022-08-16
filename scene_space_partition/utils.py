import numpy as np


class RGBD2PointCloud(object):

    def __init__(self, img_wid, img_hei, intrin):
        super(RGBD2PointCloud, self).__init__()
        self.image_width = img_wid
        self.image_height = img_hei
        self.intrinsics = intrin

    def RGBD2local(self, color, depth, filter_invalid=False): # color: [0, 1]. depth: in meter.
        cx, cy, fx, fy = self.intrinsics[0, 2], self.intrinsics[1, 2], self.intrinsics[0, 0], self.intrinsics[1, 1]
        u_base = np.tile(np.arange(self.image_width), (self.image_height, 1))
        v_base = np.tile(np.arange(self.image_height)[:, np.newaxis], (1, self.image_width))
        X = (u_base - cx) * depth / fx
        Y = (v_base - cy) * depth / fy
        coord_camera = np.stack((X, Y, depth), axis=2)
        xyz = coord_camera.reshape((-1, 3))
        xyzrgb = np.concatenate((xyz, color.reshape((-1, 3))), dtype=np.float32, axis=1)
        # xyz = coord_camera.reshape((-1, 3), order='F')
        # xyzrgb = np.concatenate((xyz, color.reshape((-1, 3), order='F')), dtype=np.float32, axis=1)
        # remove origin in the pointcloud.
        if filter_invalid: 
            s = np.sum(xyzrgb[:,0:3], axis=1)
            ids = ~(s==0)
            xyzrgb = xyzrgb[ids]
        return xyzrgb # (N, 6).

    def local2world(self, xyzrgb_local, pose):
        points_local = xyzrgb_local[:,0:3]
        points_local_homo = np.concatenate((points_local, np.ones((points_local.shape[0], 1), dtype=np.float32)), axis=1) # N*4.
        points_world_homo = np.matmul(pose, points_local_homo.T).T # (4*4 * 4*N).T = N*4.
        points_world = np.divide(points_world_homo, points_world_homo[:, [-1]])[:, :-1]
        return np.concatenate((points_world, xyzrgb_local[:,3:6]), axis=1)

    def RGBD_pose_2pc(self, color, depth, pose, filter_invalid=False):
        xyzrgb_local = self.RGBD2local(color, depth, filter_invalid)
        xyzrgb_world = self.local2world(xyzrgb_local, pose)
        return xyzrgb_world

