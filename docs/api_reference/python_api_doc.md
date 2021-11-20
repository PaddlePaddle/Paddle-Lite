
# Python API

## create\_paddle\_predictor


```python 
paddlelite.lite.create_paddle_predictor(config: MobileConfig)
```

根据 `MobileConfig` 配置构建预测器。

示例：

```Python
// 设置 MobileConfig
config = MobileConfig()
config.set_model_from_file(<your_model_path>)

// 根据 MobileConfig 创建 PaddlePredictor
predictor = create_paddle_predictor(config)
```
- 参数

    - `config`: 用于构建预测器的配置信息。

- 返回值

    - 生成的预测器对象。

## MobileConfig


```python 
class paddle_lite.lite.MobileConfig;
```

`MobileConfig` 是用来配置构建轻量级 PaddlePredictor 的配置信息，如 NaiveBuffer 格式的模型地址、能耗模式、工作线程数等。

*注意：输入的模型需要使用 [Model Optimize Tool](../user_guides/model_optimize_tool) 转化为 NaiveBuffer 格式的优化模型。*

示例：

```python 
config = MobileConfig()

// 设置 NaiveBuffer 格式模型目录，从文件加载模型时使用
config.set_model_from_file(<your_model_path>)

// 设置工作线程数
config.set_threads(4)

// 设置能耗模式
config.set_power_mode(LITE_POWER_HIGH)

// 根据 MobileConfig 创建 PaddlePredictor
predictor = create_paddle_predictor(config)
```

### `set_model_from_file`

```python
MobileConfig.set_model_from_file(x: str)
```

设置模型文件，当需要从磁盘加载模型时使用。

- 参数

    - `x`: 模型文件路径。

### `set_model_dir`

```python
MobileConfig.set_model_dir(x: str)
```

**注意**：Lite 模型格式在 release/v2.3.0 之后修改，本接口为加载老格式模型的接口，将在 release/v3.0.0 废弃。建议替换为 `set_model_from_file` 接口。

设置模型文件夹路径，当需要从磁盘加载模型时使用。

- 参数

    - `x`: 模型文件夹路径

### `model_dir`

```python
MobileConfig.model_dir()
```
获取设置的模型文件夹路径。

- 返回值

    - 模型文件夹路径。

### `set_power_mode`

```python
MobileConfig.set_power_mode(mode: PowerMode)
```

设置 CPU 能耗模式。若不设置，则默认使用 `LITE_POWER_HIGH`。

*注意：只在开启 `OpenMP` 时生效，否则系统自动调度。*

- 参数

    - `PowerMode`: CPU 能耗模式。

### `power_mode`

```python
MobileConfig.power_mode()
```

获取设置的 CPU 能耗模式。

- 返回值

    - 设置的 CPU 能耗模式


### `set_threads`

```python
MobileConfig.set_threads(threads: int)
```

设置工作线程数。若不设置，则默认使用单线程。

*注意：只在开启 `OpenMP` 的模式下生效，否则只使用单线程。*

- 参数

    - `threads`: 工作线程数。

### `threads`

```python
MobileConfig.threads()
```

获取设置的工作线程数。

- 返回值

    - 工作线程数。


## PaddlePredictor


```python
class paddle_lite.lite.PaddlePredictor
```

`PaddlePredictor` 是 Paddle Lite 的预测器，由 `create_paddle_predictor` 根据 `MobileConfig` 进行创建。用户可以根据 PaddlePredictor 提供的接口设置输入数据、执行模型预测、获取输出等。

示例：

```python
from paddlelite.lite import *
import numpy as np
from PIL import Image

# (1) 设置配置信息
config = MobileConfig()
config.set_model_from_file("./mobilenet_v1_opt.nb")

# (2) 创建预测器
predictor = create_paddle_predictor(config)

# (3) 从图片读入数据
image = Image.open('./example.jpg')
resized_image = image.resize((224, 224), Image.BILINEAR)
image_data = np.array(resized_image).transpose(2, 0, 1).reshape(1, 3, 224, 224)

# (4) 设置输入数据
input_tensor = predictor.get_input(0)
input_tensor.from_numpy(image_data)

# (5) 执行预测
predictor.run()

# (6) 得到输出数据
output_tensor = predictor.get_output(0)
print(output_tensor.shape())
print(output_tensor.numpy())
```

### `get_input`

```python
PaddlePredictor.get_input(i: int)
```

获取输入 Tensor 的引用，用来设置模型的输入数据。

- 参数

    - `i`: 输入 Tensor 的索引。

- 返回值

    - 第 `i` 个输入 Tensor 的引用。

### `get_output`

```python
PaddlePredictor.get_output(i: int)
```

获取输出 Tensor 的引用，用来获取模型的输出结果。

- 参数

    - `i`: 输出 Tensor 的索引。

- 返回值

    - 第 `i` 个输出 Tensor 的引用。


### `run`

```python
PaddlePredictor.run()
```

执行模型预测，需要在 ***设置输入数据后*** 调用。

### `get_version`

```python
PaddlePredictor.get_version()
```

获取输出 Tensor 的引用，用来获取模型的输出结果。

用于获取当前库使用的代码版本。若代码有相应标签则返回标签信息，如 v2.0-beta；否则返回代码的 branch (commit id)，如 develop (7e44619)。

- 返回值

    - 当前库使用的代码版本信息。



## TargetType


```python
enum paddle_lite.lite.TargetType
```
`TargetType` 为目标设备硬件类型，用户可以根据应用场景选择硬件平台类型。

枚举型变量 `TargetType` 的取值包括 `{X86, CUDA, ARM, OpenCL, FPGA, NPU}` 等。

## PrecisionType

```python
enum paddle_lite.lite.PrecisionType
```
`PrecisionType` 为模型中 Tensor 的数据精度，默认值为 FP32 (float32)。

枚举型变量 `PrecisionType` 的取值包括 `{FP32, INT8, INT32, INT64}` 等。

## DataLayoutType

```python
enum paddle_lite.lite.DataLayoutType
```
`DataLayoutType` 为 Tensor 的数据格式，默认值为 NCHW（number, channel, height, weigth）。

枚举型变量 `DataLayoutType` 的取值包括 `{NCHW, NHWC}` 等。

## Place

```python
class paddle_lite.lite.Place
```

`Place` 是 `TargetType`、`PrecisionType` 和 `DataLayoutType` 的集合，说明运行时的设备类型、数据精度和数据格式。


## PowerMode

```python
enum paddle_lite.lite.PowerMode
```

`PowerMode` 为 ARM CPU 能耗模式，用户可以根据应用场景设置能耗模式获得最优的能效比。


|         选项         | 说明                                                         |
| :------------------: | ------------------------------------------------------------ |
|   LITE\_POWER\_HIGH    | 绑定大核运行模式。如果 ARM CPU 支持 big.LITTLE，则优先使用并绑定 Big cluster，如果设置的线程数大于大核数量，则会将线程数自动缩放到大核数量。如果系统不存在大核或者在一些手机的低电量情况下会出现绑核失败，如果失败则进入不绑核模式。 |
|    LITE\_POWER\_LOW    | 绑定小核运行模式。如果 ARM CPU 支持 big.LITTLE，则优先使用并绑定 Little cluster，如果设置的线程数大于小核数量，则会将线程数自动缩放到小核数量。如果找不到小核，则自动进入不绑核模式。 |
|   LITE\_POWER\_FULL    | 大小核混用模式。线程数可以大于大核数量，当线程数大于核心数量时，则会自动将线程数缩放到核心数量。 |
|  LITE\_POWER\_NO\_BIND  | 不绑核运行模式（推荐）。系统根据负载自动调度任务到空闲的 CPU 核心上。 |
| LITE\_POWER\_RAND\_HIGH | 轮流绑定大核模式。如果 Big cluster 有多个核心，则每预测 10 次后切换绑定到下一个核心。 |
| LITE\_POWER\_RAND\_LOW  | 轮流绑定小核模式。如果 Little cluster 有多个核心，则每预测 10 次后切换绑定到下一个核心。 |


## Tensor


```python
class paddle_lite.lite.Tensor
```

Tensor 是 Paddle Lite 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、数据、LoD 信息等。

*注意：用户应使用 `PaddlePredictor` 的 `get_input` 和 `get_output` 接口获取输入 / 输出的`Tensor`。*


### `resize`

```python
Tensor.resize(shape: list[int])
```

设置 Tensor 的维度信息。

- 参数

    - `shape`：维度信息。

### `shape`

```python
Tensor.shape()
```

获取 Tensor 的维度信息。

- 返回值

    - Tensor 的维度信息。


### `set_lod`

```python
Tensor.set_lod(lod: list[list[int]])
```

设置 Tensor 的 LoD 信息。

- 参数

    - `lod`: Tensor 的 LoD 信息，类型为二维的 `list`。

### `lod`

```python
Tensor.lod()
```

获取 Tensor 的 LoD 信息。

- 返回值

    - Tensor 的 LoD 信息，类型为二维的 `list`。


### `precision`

```python
Tensor.precision()
```

获取 Tensor 的精度信息

- 返回值

    - `Tensor` 的精度信息，类型为 `PrecisionType`。


### `target`

```python
Tensor.target()
```

获取 Tensor 的数据所处设备信息。

- 返回值

    - `Tensor` 的数据所处设备信息。
