## TargetType

```python
class TargetType;
```
`TargetType`为目标设备硬件类型，用户可以根据应用场景选择硬件平台类型。

枚举型变量`TargetType`的所有可能取值包括：

`{X86, CUDA, ARM, OpenCL, FPGA, NPU}`


## PrecisionType
```python
class PrecisionType {FP32};
```
`PrecisionType`为模型中Tensor的数据精度，默认值为FP32(float32)。

枚举型变量`PrecisionType`的所有可能取值包括：

`{FP32, INT8, INT32, INT64}`




## DataLayoutType

```python
class DataLayoutType {NCHW};
```
`DataLayoutType`为Tensor的数据格式，默认值为NCHW（number, channel, height, weigth）。

枚举型变量`DataLayoutType`的所有可能取值包括：

` {NCHW, NHWC}`



## Place
```python
class Place{
  TargetType target;
  PrecisionType precision{FP32};
  DataLayoutType layout{NCHW}
}
```
`Place`是`TargetType`、`PrecisionType`和`DataLayoutType`的集合，说明运行时的设备类型、数据精度和数据格式。

示例：
```python
from paddlelite.lite import *

Place{TargetType(ARM), PrecisionType(FP32), DataLayoutType(NCHW)}
```
