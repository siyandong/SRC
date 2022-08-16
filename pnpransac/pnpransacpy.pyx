# distutils: language = c++

import numpy as np
cimport numpy as np

from pnpransacpy cimport PnPRANSAC
                     
cdef class pnpransac:
    cdef PnPRANSAC c_pnpransac
    
    def __cinit__(self, float fx, float fy, float cx, float cy):
        self.c_pnpransac = PnPRANSAC(fx, fy, cx, cy)

    def update_camMat(self, float fx, float fy, float cx, float cy):
        self.c_pnpransac.camMatUpdate(fx, fy, cx, cy)

    def RANSAC_one2many(self, 
                 np.ndarray[double, ndim=2, mode="c"] img_pts, 
                 np.ndarray[double, ndim=3, mode="c"] obj_pts, 
                 int n_hyp):
        #print(obj_pts[66][3])
        cdef float[:, :] img_pts_ = img_pts.astype(np.float32)
        cdef float[:, :, :] obj_pts_ = obj_pts.astype(np.float32)
        cdef int n_pts, n_cps
        n_pts, n_cps = img_pts_.shape[0], obj_pts_.shape[1]
        assert img_pts_.shape[0] == obj_pts_.shape[0]
        cdef double* pose
        pose = self.c_pnpransac.RANSAC_one2many(&img_pts_[0,0], &obj_pts_[0,0,0], n_pts, n_cps, n_hyp)
        rot =  np.array([pose[0],pose[1],pose[2]])
        transl = np.array([pose[3],pose[4],pose[5]])
        return rot, transl

