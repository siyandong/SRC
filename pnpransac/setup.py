import sys
import os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

# where to find opencv headers and libraries
cv_include_dir = os.path.join(sys.prefix, 'include')
cv_library_dir = os.path.join(sys.prefix, 'lib')
#print('using opencv from path\ninclude: {}\nlib: {}\n'.format(cv_include_dir, cv_library_dir))

ext_modules = [
    Extension(
        "pnpransac",
        sources=["pnpransacpy.pyx"],
        language="c++",
        include_dirs=[cv_include_dir, np.get_include()],
        library_dirs=[cv_library_dir],
        # include_dirs=[
        #             '/local/home/sidong/Desktop/code/opencv/opencv-4.x/include',
        #             '/local/home/sidong/Desktop/code/opencv_contrib/build',
        #             np.get_include()],
        # library_dirs=[
        #             '/local/home/sidong/Desktop/code/opencv/build/lib',
        #             '/local/home/sidong/Desktop/code/opencv_contrib/build/lib',
        #             ],
        libraries=['opencv_core','opencv_calib3d'],
        extra_compile_args=['-fopenmp','-std=c++11'],
    )
]

setup(
    name='pnpransac',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules),
    )