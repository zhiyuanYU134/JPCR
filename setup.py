from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='PCR_Jittor',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='PCR_Jittor.ext',
            sources=[
                'PCR_Jittor/extensions/extra/cloud/cloud.cpp',
                'PCR_Jittor/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'PCR_Jittor/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'PCR_Jittor/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'PCR_Jittor/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'PCR_Jittor/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
