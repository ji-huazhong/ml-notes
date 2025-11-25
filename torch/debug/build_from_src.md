## # Ubuntu 22.04 源码编译PyTorch

## 基础环境

使用 Anaconda 维护 PyTorch 环境。

```shell
conda create --name torch271 python=3.10
conda activate torch271
conda deactivate
conda remove --name torch271 --all
conda create --name <newenv> --clone <oldenv>
```



## Torch 安装

1. 安装依赖
   
   ```shell
   conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
   ```

2. 下载 PyTorch v2.5.1 源码
   
   ```shell
   git clone -b v2.5.1 https://github.com/pytorch/pytorch torch_v2.5.1
   cd torch_v2.5.1
   git submodule sync
   git submodule update --init --recursive
   ```

3. 使用 CCache 提高编译速度
   
   ```shell
   conda install ccache
   # 设置ccache缓存最大大小为 25GiB
   ccache -M 25Gis
   # 设置缓存文件数量为无限制
   ccache -F 0
   # 使用 ccache 作为编译器启动器，从而启动 ccache 缓存功能
   export CMAKE_C_COMPILE_LAUNCHER=ccache
   export CMAKE_CXX_COMPILE_LAUNCHER=ccache
   export CMAKE_CUDA_COMPILE_LAUNCHER=ccache
   ```
   
   缓存目录为 `~/.ccache`，配置文件为 `~/.ccache/ccache.conf`

4. 编译 Torch CPU 调试版本

   ```
   DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_CUDA=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 python setup.py develop
   ```

5. 如果编译 Torch GPU 版本

   ```shell
   DEBUG=1 USE_MKLDNN=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 python setup.py develop
   ```

## 参考

- [Ubuntu 22.04 LTS 源码编译安装 PyTorch_pytroch编译 安装-CSDN博客](https://blog.csdn.net/weixin_43254181/article/details/135554512)
- https://xugaoxiang.com/2020/11/20/build-pytorch-with-gpu-from-source/

