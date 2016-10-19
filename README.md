# Dense and Sparse Labeling (DSL) with Multi-Dimensional Features for Saliency Detection
This is the software for paper [1]. Please cite [1] if you use this code.
Author: [Yuchen Yuan](mailto:yuchen.yuan@sydney.edu.au)
Last updated: Oct 18, 2016

## Installation
This software is implemented on MatConvNet [2] with CUDA 7.5 and cuDNN v3. CPU-only mode is also supported.
- **Resources**: The model files and already-generated saliency maps on existing datasets can be downloaded [here](http://pan.baidu.com/s/1c2hzzKc)
- **Supported OS**: This software is tested on 64-bit Ubuntu 14.04 and 64-bit Windows 8.1.
- **MatConvNet**: Please download [MatConvNet](http://www.vlfeat.org/matconvnet/) to the current path, and compile with [instructions](http://www.vlfeat.org/matconvnet/install/). Below is a compilation example:
```
run matlab/vl_setupnn.m
vl_compilenn('enableGpu', true, 'cudaMethod', 'nvcc', ...
'cudaRoot', '/usr/local/cuda-7.5', ...
'enableCudnn', true, 'cudnnRoot', '/usr/local/cuda/');
```
- **CUDA**: If run with GPU, please download and install [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
- **cuDNN**: If run with GPU, please download and install [cuDNN](https://developer.nvidia.com/cudnn)
- **wine**: If run under Linux, please install [wine](sudo apt-get install wine) for the SLIC program support.

## Usage
- **Entrance**: Please run `dsl_demo.m` for an example use. 
- **Default input image path**: `image`.
- **Default trained network path**: `model`.
- **Default result path**: `result/1_DL` for FCN results, `result/2_SL` for CCN results, and `result/3_DC` for DCN (final saliency map) results.
- **GPU or CPU mode**: Please set `gpus = 1` for GPU mode, or `gpus = []` for CPU-only mode.

## Notes
- If an error in `dagnn.BatchNorm` occurs, please replace `matconvnet/matlab/+dagnn/BatchNorm.m` with `support/BatchNorm.m`
- If an error in `dagnn.ReLU` occurs, please replace `matconvnet/matlab/+dagnn/ReLU.m` with `support/ReLU.m`
- If a `mex_link` error is encountered while compiling MatConvNet, please try replacing the "parfor" with "for" in `vl_compilenn.m`. This issue is fixed in the latest version of MatConvNet.

## References
> [1] Y. Yuan, C. Li *et al.* "Dense and sparse labeling with multi-dimensional features for saliency detection", *IEEE Trans. Circuits and Syst. Video Technol.*, vol. xx, no. xx, pp. xx-yy, Month. 2016
> [2] A. Vedaldi and K. Lenc, "MatConvNet-convolutional neural networks for MATLAB", *arXiv preprint arXiv*:1412.4564, 2014.
