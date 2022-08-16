import numpy as np
import open3d as o3d
from easydict import EasyDict as edict


opt = edict()
opt.pixel_step = 4 # useless?
opt.image_width, opt.image_height = 640, 480
opt.intrinsics = np.array([[525, 0, 320],[0, 525, 240],[0, 0, 1]], dtype=float)


class Renderer():
    def __init__(self, geometry, opt):
        self.geometry = geometry
        self.w = (opt.image_width - 1) // opt.pixel_step + 1
        self.h = (opt.image_height - 1) // opt.pixel_step + 1
        image_width = int((opt.intrinsics[0, 2] if opt.intrinsics[0, 2] * 2 > opt.image_width
                     else opt.image_width - opt.intrinsics[0, 2])
                    * 2 / opt.pixel_step)
        image_height = int((opt.intrinsics[1, 2] if opt.intrinsics[1, 2] * 2 > opt.image_height
                      else opt.image_height - opt.intrinsics[1, 2])
                     * 2 / opt.pixel_step)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=image_width, height=image_height, visible=False)
        self.vis.add_geometry(self.geometry)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.get_render_option().mesh_show_back_face = True
        # self.vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
        self.ctr = self.vis.get_view_control()
        self.param = self.ctr.convert_to_pinhole_camera_parameters()
        self.param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            0, 0,
            0, opt.intrinsics[1, 1] / opt.pixel_step,
            0, 0)

    # def __del__(self):
    #     self.vis.clear_geometries()

    def depth(self, pose):
        self.param.extrinsic = np.linalg.inv(pose)
        self.ctr.convert_from_pinhole_camera_parameters(self.param, True)
        self.vis.poll_events()
        self.vis.update_renderer()
        depth = np.asarray(self.vis.capture_depth_float_buffer(False))
        if opt.intrinsics[1, 2] * 2 > image_height:
            depth = depth[:self.h, :]
        else:
            depth = depth[depth.shape[0] - self.h:, :]
        if opt.intrinsics[0, 2] * 2 > image_width:
            depth = depth[:, :self.w]
        else:
            depth = depth[:, depth.shape[1] - self.w:]
        return depth

    def color(self, pose):
        self.param.extrinsic = np.linalg.inv(pose)
        self.ctr.convert_from_pinhole_camera_parameters(self.param, True)
        self.vis.poll_events()
        self.vis.update_renderer()
        color = np.asarray(self.vis.capture_screen_float_buffer(False))
        if opt.intrinsics[1, 2] * 2 > image_height:
            color = color[:self.h, :]
        else:
            color = color[color.shape[0] - self.h:, :]
        if opt.intrinsics[0, 2] * 2 > image_width:
            color = color[:, :self.w]
        else:
            color = color[:, color.shape[1] - self.w:]
        return color


if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh('_.ply')
    renderer = Renderer(mesh, opt)



