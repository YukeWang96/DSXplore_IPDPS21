from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='scc',
    ext_modules=[
        CUDAExtension('scc_cuda', [
            'scc_cuda.cpp',
            'scc_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })