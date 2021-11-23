# 使用 GPU 获取最佳性能

Paddle Lite 支持在 Android 和 iOS 等设备上使用 GPU 后端获取最佳的性能。

GPU 可以用来运行较大运算强度的负载任务，将模型中计算任务切分为更小的工作负载，利用 GPU 提供的大规模线程来并行工作，从而获取高吞吐和较低延迟。与 CPU 不同，GPU 支持运行在 32 位浮点模式或者 16 位浮点模式而不用通过量化来获得最佳的性能。

Paddle Lite 支持多种 GPU 后端，包括 OpenCL、[Metal](https://developer.apple.com/metal/), 支持包括 ARM Mali、Qualcomm Adreno、Apple A Series 等系列 GPU 设备。

## Android 设备使用 OpenCL 获取最佳性能
详细见[OpenCL 部署示例](opencl.md)

## iOS 设备使用 Metal 获取最佳性能


## 支持的模型与Ops
GPU 支持的模型列表见[支持模型](../quick_start/support_model_list), 详细的 OP 支持列表见[支持算子](quick_start/support_operation_list).

## 优化建议
* 减少低计算量、高访存算子的使用，如 concat、slice 等。
* 尽量减少只改变形状而没有计算量的算子使用，如 reshape、transpose、permute 等。
* 不宜使用太大尺寸的卷积核，尽量使用最常用的尺寸为 1x1 和 3x3 的卷积核。当模型需要较大卷积核时，可以考虑使用小卷积核进行代替。