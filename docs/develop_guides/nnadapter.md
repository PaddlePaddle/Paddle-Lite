# NNAdapter
飞桨推理AI硬件统一适配框架

## 背景
- 在[新增硬件](./add_hardware)章节中曾提到Paddle Lite的硬件接入分为算子Kernel和子图两种方式

## 简介
- 

## 实现方案

## 功能模块

### NNAdapter API
### NNAdapter 标准算子
### NNAdapter Runtime
### NNAdapter HAL标准接口定义
- 示例

## 推理框架、NNAdapter和硬件SDK的调用关系

## 附录

### NNAdapter API详细说明
- NNAdapter_getVersion
  ```c++
  int NNAdapter_getVersion(uint32_t* version)
  ```
  获取NNAdapter版本值。
  - 参数：
    - version：存储返回NNAdapter的版本值。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。
- NNAdapterDevice_acquire
  ```c++
  NNAdapterDevice_acquire(const char* name, NNAdapterDevice** device)
  ```
  通过名称获取设备实例。
  - 参数：
    - name：通过该名称加载并注册设备HAL库后（仅发生在进程首次调用时），创建一个设备实例。
    - device：存储创建后的设备实例。
  - 返回值：调用成功则返回NNADAPTER_NO_ERROR。

### NNAdapter 标准算子详细说明
- NNADAPTER_ABS

  Applies the abs activation to the input tensor element-wise. The output is calculated using this formula: output = abs(input)
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
  - Outputs:
    - 0: output, the result with the same type as two inputs.

- NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D

  Applies adaptive 2-D average pooling across the input according to input and output size.
  - Inputs:
    - 0: input, a NNADAPTER_TENSOR_FLOAT32, NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in, H_in, W_in].
    - 1: output_shape, a NNADAPTER_TENSOR_INT32 or NNADAPTER_TENSOR_INT64 tensor, with shape [2], with value [H_out, H_out].
  - Outputs:
    - 0: output, a tensor with the same shape and type as input.
