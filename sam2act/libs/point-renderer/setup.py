# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys
from setuptools import setup, find_packages, dist
import glob
import logging

PACKAGE_NAME = 'point_renderer'
DESCRIPTION = 'Fast Point Cloud Renderer'
URL = 'https://gitlab-master.nvidia.com/vblukis/point-renderer'
AUTHOR = 'Valts Blukis'
LICENSE = 'NVIDIA'
version = '0.2.0'


def get_extensions():
    # 延迟导入 torch，避免在构建依赖安装前就需要它
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    except ImportError:
        # 如果 torch 不可用，返回 None，让 setuptools 处理依赖
        return None
    
    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []
    include_dirs = []
    extensions = []
    sources = glob.glob('point_renderer/csrc/**/*.cpp', recursive=True)

    if len(sources) == 0:
        print("No source files found for extension, skipping extension compilation")
        return None

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('point_renderer/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args.update({'nvcc': ['-O3']})
        #include_dirs = get_include_dirs()
    else:
        assert False, "CUDA is not available. Set FORCE_CUDA=1 for Docker builds"

    extensions.append(
        extension(
            name='point_renderer._C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            #include_dirs=include_dirs
        )
    )

    for ext in extensions:
        ext.libraries = ['cudart_static' if x == 'cudart' else x
                         for x in ext.libraries]

    return extensions


if __name__ == '__main__':
    # 延迟导入 BuildExtension
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension
        
        # 创建一个自定义的 BuildExtension 来绕过 CUDA 版本检查
        class CustomBuildExtension(BuildExtension):
            def build_extensions(self):
                # 临时禁用 CUDA 版本检查
                import torch.utils.cpp_extension as cpp_ext
                original_check = cpp_ext._check_cuda_version
                def noop_check(*args, **kwargs):
                    pass
                cpp_ext._check_cuda_version = noop_check
                try:
                    super().build_extensions()
                finally:
                    # 恢复原始检查函数
                    cpp_ext._check_cuda_version = original_check
        
        cmdclass = {
            'build_ext': CustomBuildExtension.with_options(no_python_abi_suffix=True)
        }
    except ImportError:
        cmdclass = {}
    
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=version,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        license=LICENSE,
        python_requires='>=3.7',

        # Package info
        packages=['point_renderer'],
        include_package_data=True,
        zip_safe=True,
        ext_modules=get_extensions() or [],
        cmdclass=cmdclass

    )
