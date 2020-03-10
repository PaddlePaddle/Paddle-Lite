# Python API

## create_paddle_predictor

```python
CxxPredictor create_paddle_predictor(config); # config为CxxConfig类型
LightPredictor create_paddle_predictor(config); # config为MobileConfig类型
```

`create_paddle_predictor`函数用来根据`CxxConfig`或`MobileConfig`构建预测器。

示例：

```python
from lite_core import *

# 设置CxxConfig
config = CxxConfig()
config.set_model_dir(<your_model_dir_path>)
places = [Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 根据CxxConfig创建CxxPredictor
predictor = create_paddle_predictor(config)
```

参数：

- `config(CxxConfig或MobileConfig)` - 用于构建Predictor的配置信息。

返回：预测器`predictor`

返回类型：`CxxPredictor`或`LightPredictor`

## CxxConfig

```python
class CxxConfig;
```

`CxxConfig`用来配置构建CxxPredictor的配置信息，如protobuf格式的模型地址、能耗模式、工作线程数、place信息等等。

示例：

```python
from lite_core import *

config = CxxConfig()
# 设置模型目录，加载非combined模型时使用
config.set_model_dir(<your_model_dir_path>)
# 设置工作线程数
config.set_threads(4);
# 设置能耗模式
config.set_power_mode(PowerMode.LITE_POWER_NO_BIND)
# 设置valid places
places = [Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 根据CxxConfig创建CxxPredictor
predictor = create_paddle_predictor(config)
```

### `set_model_dir(model_dir)`

设置模型文件夹路径，当需要从磁盘加载非combined模型时使用。

参数：

- `model_dir(str)` - 模型文件夹路径

返回：`None`

返回类型：`None`



### `model_dir()`

返回设置的模型文件夹路径。

参数：

- `None`

返回：模型文件夹路径

返回类型：`str`



### `set_model_file(model_file)`

设置模型文件路径，加载combined形式模型时使用。

参数：

- `model_file(str)` - 模型文件路径

返回类型：`None`



### `model_file()`

获取设置模型文件路径，加载combined形式模型时使用。

参数：

- `None`

返回：模型文件路径

返回类型：`str`



### `set_param_file(param_file)`

设置模型参数文件路径，加载combined形式模型时使用。

参数：

- `param_file(str)` - 模型文件路径

返回类型：`None`



### `param_file()`

获取设置模型参数文件路径，加载combined形式模型时使用。

参数：

- `None`

返回：模型参数文件路径

返回类型：`str`



### `set_valid_places(valid_places)`

设置可用的places列表。

参数：

- `valid_places(list)` - 可用place列表。

返回类型：`None`

示例：

```python
from lite_core import *

config = CxxConfig()
# 设置模型目录，加载非combined模型时使用
config.set_model_dir(<your_model_dir_path>)
# 设置valid places
# 注意，valid_places列表中Place的排序表明了用户对Place的偏好程度，如用户想优先使用ARM上Int8精度的
# kernel，则应把Place(TargetType.ARM, PrecisionType.INT8)置于valid_places列表的首位。
places = [Place(TargetType.ARM, PrecisionType.INT8),
          Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 根据CxxConfig创建CxxPredictor
predictor = create_paddle_predictor(config)
```



### `set_power_mode(mode)`

设置CPU能耗模式。若不设置，则默认使用`PowerMode.LITE_POWER_HIGH`。

*注意：只在开启`OpenMP`时生效，否则系统自动调度。此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `mode(PowerMode)` - CPU能耗模式

返回：`None`

返回类型：`None`



### `power_mode()`

获取设置的CPU能耗模式。

*注意：此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `None`

返回：设置的CPU能耗模式

返回类型：`PowerMode`



### `set_threads(threads)`

设置工作线程数。若不设置，则默认使用单线程。

*注意：只在开启`OpenMP`的模式下生效，否则只使用单线程。此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `threads(int)` - 工作线程数

返回：`None`

返回类型：`None`



### `threads()`

获取设置的工作线程数。

*注意：此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `None`

返回：工作线程数

返回类型：`int`

## MobileConfig

```python
class MobileConfig;
```

`MobileConfig`用来配置构建LightPredictor的配置信息，如NaiveBuffer格式的模型地址、能耗模式、工作线程数等等。

示例：

```python
from lite_core import *

config = MobileConfig()
# 设置NaiveBuffer格式模型目录
config.set_model_from_file(<your_model_path>)
# 设置工作线程数
config.set_threads(4);
# 设置能耗模式
config.set_power_mode(PowerMode.LITE_POWER_NO_BIND)

# 根据MobileConfig创建LightPredictor
predictor = create_paddle_predictor(config)
```

### `set_model_from_file(model_file)`

**注意**：`model_file`应该是经过`opt`优化后产生的`NaiveBuffer`格式的模型。

设置模型文件夹路径。

参数：

- `model_file(str)` - 模型文件路径

返回：`None`

返回类型：`None`



### `set_model_dir(model_dir)`

**注意**：Lite模型格式在release/v2.3.0之后修改，本接口为加载老格式模型的接口，将在release/v3.0.0废弃。建议替换为`setModelFromFile`接口。`model_dir`应该是经过`Model Optimize Tool`优化后产生的`NaiveBuffer`格式的模型。

设置模型文件夹路径。

参数：

- `model_dir(str)` - 模型文件夹路径

返回：`None`

返回类型：`None`



### `set_model_from_buffer(model_buffer)`

设置模型的内存数据，当需要从内存加载模型时使用。

参数：

- `model_buffer(str)` - 内存中的模型数据

返回：`None`

返回类型：`void`




### `model_dir()`

返回设置的模型文件夹路径。

参数：

- `None`

返回：模型文件夹路径

返回类型：`str`



### `set_power_mode(mode)`

设置CPU能耗模式。若不设置，则默认使用`PowerMode.LITE_POWER_HIGH`。

*注意：只在开启`OpenMP`时生效，否则系统自动调度。此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `mode(PowerMode)` - CPU能耗模式

返回：`None`

返回类型：`None`



### `power_mode()`

获取设置的CPU能耗模式。

*注意：此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `None`

返回：设置的CPU能耗模式

返回类型：`PowerMode`



### `set_threads(threads)`

设置工作线程数。若不设置，则默认使用单线程。

*注意：只在开启`OpenMP`的模式下生效，否则只使用单线程。此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `threads(int)` - 工作线程数

返回：`None`

返回类型：`None`



### `threads()`

获取设置的工作线程数。

*注意：此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `None`

返回：工作线程数

返回类型：`int`

## CxxPredictor

```c++
class CxxPredictor
```

`CxxPredictor`是Paddle-Lite的预测器，由`create_paddle_predictor`根据`CxxConfig`进行创建。用户可以根据CxxPredictor提供的接口设置输入数据、执行模型预测、获取输出以及获得当前使用lib的版本信息等。

示例：

```python
from __future__ import print_function
from lite_core import *

# 1. 设置CxxConfig
config = CxxConfig()
if args.model_file != '' and args.param_file != '':
    config.set_model_file(args.model_file)
    config.set_param_file(args.param_file)
else:
    config.set_model_dir(args.model_dir)
places = [Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 2. 创建CxxPredictor
predictor = create_paddle_predictor(config)

# 3. 设置输入数据
input_tensor = predictor.get_input(0)
input_tensor.resize([1, 3, 224, 224])
input_tensor.set_float_data([1.] * 3 * 224 * 224)

# 4. 运行模型
predictor.run()

# 5. 获取输出数据
output_tensor = predictor.get_output(0)
print(output_tensor.shape())
print(output_tensor.float_data()[:10])
```

### `get_input(index)`

获取输入Tensor，用来设置模型的输入数据。

参数：

- `index(int)` - 输入Tensor的索引

返回：第`index`个输入`Tensor`

返回类型：`Tensor`



### `get_output(index)`

获取输出Tensor，用来获取模型的输出结果。

参数：

- `index(int)` - 输出Tensor的索引

返回：第`index`个输出`Tensor`

返回类型：`Tensor`



### `run()`

执行模型预测，需要在***设置输入数据后***调用。

参数：

- `None`

返回：`None`

返回类型：`None`



### `get_version()`

用于获取当前lib使用的代码版本。若代码有相应tag则返回tag信息，如`v2.0-beta`；否则返回代码的`branch(commitid)`，如`develop(7e44619)`。

参数：

- `None`

返回：当前lib使用的代码版本信息

返回类型：`str`

## LightPredictor

```c++
class LightPredictor
```

`LightPredictor`是Paddle-Lite的预测器，由`create_paddle_predictor`根据`MobileConfig`进行创建。用户可以根据LightPredictor提供的接口设置输入数据、执行模型预测、获取输出以及获得当前使用lib的版本信息等。

示例：

```python
from __future__ import print_function
from lite_core import *

# 1. 设置MobileConfig
config = MobileConfig()
config.set_model_dir(args.model_dir)

# 2. 创建LightPredictor
predictor = create_paddle_predictor(config)

# 3. 设置输入数据
input_tensor = predictor.get_input(0)
input_tensor.resize([1, 3, 224, 224])
input_tensor.set_float_data([1.] * 3 * 224 * 224)

# 4. 运行模型
predictor.run()

# 5. 获取输出数据
output_tensor = predictor.get_output(0)
print(output_tensor.shape())
print(output_tensor.float_data()[:10])
```

### `get_input(index)`

获取输入Tensor，用来设置模型的输入数据。

参数：

- `index(int)` - 输入Tensor的索引

返回：第`index`个输入`Tensor`

返回类型：`Tensor`



### `get_output(index)`

获取输出Tensor，用来获取模型的输出结果。

参数：

- `index(int)` - 输出Tensor的索引

返回：第`index`个输出`Tensor`

返回类型：`Tensor`



### `run()`

执行模型预测，需要在***设置输入数据后***调用。

参数：

- `None`

返回：`None`

返回类型：`None`



### `get_version()`

用于获取当前lib使用的代码版本。若代码有相应tag则返回tag信息，如`v2.0-beta`；否则返回代码的`branch(commitid)`，如`develop(7e44619)`。

参数：

- `None`

返回：当前lib使用的代码版本信息

返回类型：`str`

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
from lite_core import *

Place{TargetType(ARM), PrecisionType(FP32), DataLayoutType(NCHW)}
```



## PowerMode

```python
class PowerMode;
```

`PowerMode`为ARM CPU能耗模式，用户可以根据应用场景设置能耗模式获得最优的能效比。

示例：

```python
from lite_core import *

config = MobileConfig()
# 设置NaiveBuffer格式模型目录
config.set_model_dir(<your_model_dir_path>)
# 设置能耗模式
config.set_power_mode(PowerMode.LITE_POWER_NO_BIND)

# 根据MobileConfig创建LightPredictor
predictor = create_paddle_predictor(config)
```

PowerMode详细说明如下：

|         选项         | 说明                                                         |
| :------------------: | ------------------------------------------------------------ |
|   LITE_POWER_HIGH    | 绑定大核运行模式。如果ARM CPU支持big.LITTLE，则优先使用并绑定Big cluster。如果设置的线程数大于大核数量，则会将线程数自动缩放到大核数量。如果系统不存在大核或者在一些手机的低电量情况下会出现绑核失败，如果失败则进入不绑核模式。 |
|    LITE_POWER_LOW    | 绑定小核运行模式。如果ARM CPU支持big.LITTLE，则优先使用并绑定Little cluster。如果设置的线程数大于小核数量，则会将线程数自动缩放到小核数量。如果找不到小核，则自动进入不绑核模式。 |
|   LITE_POWER_FULL    | 大小核混用模式。线程数可以大于大核数量。当线程数大于核心数量时，则会自动将线程数缩放到核心数量。 |
|  LITE_POWER_NO_BIND  | 不绑核运行模式（推荐）。系统根据负载自动调度任务到空闲的CPU核心上。 |
| LITE_POWER_RAND_HIGH | 轮流绑定大核模式。如果Big cluster有多个核心，则每预测10次后切换绑定到下一个核心。 |
| LITE_POWER_RAND_LOW  | 轮流绑定小核模式。如果Little cluster有多个核心，则每预测10次后切换绑定到下一个核心。 |



## Tensor

```c++
class Tensor
```

Tensor是Paddle-Lite的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置Shape、数据、LoD信息等。

*注意：用户应使用`CxxPredictor`或`LightPredictor`的`get_input`和`get_output`接口获取输入/输出的`Tensor`。*

示例：

```python
from __future__ import print_function
from lite_core import *

# 1. 设置CxxConfig
config = CxxConfig()
if args.model_file != '' and args.param_file != '':
    config.set_model_file(args.model_file)
    config.set_param_file(args.param_file)
else:
    config.set_model_dir(args.model_dir)
places = [Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 2. 创建CxxPredictor
predictor = create_paddle_predictor(config)

# 3. 设置输入数据
input_tensor = predictor.get_input(0)
input_tensor.resize([1, 3, 224, 224])
input_tensor.set_float_data([1.] * 3 * 224 * 224)

# 4. 运行模型
predictor.run()

# 5. 获取输出数据
output_tensor = predictor.get_output(0)
print(output_tensor.shape())
print(output_tensor.float_data()[:10])
```

### `resize(shape)`

设置Tensor的维度信息。

参数：

- `shape(list)` - 维度信息

返回：`None`

返回类型：`None`



### `shape()`

获取Tensor的维度信息。

参数：

- `None`

返回：Tensor的维度信息

返回类型：`list`



### `float_data()`

获取Tensor的持有的float型数据。

示例：

```python
output_tensor = predictor.get_output(0)
print(output_tensor.shape())
print(output_tensor.float_data()[:10])
```

参数：

- `None`

返回：`Tensor`持有的float型数据

返回类型：`list`



### `set_float_data(float_data)`

设置Tensor持有float数据。

示例：

```python
input_tensor = predictor.get_input(0)
input_tensor.resize([1, 3, 224, 224])
input_tensor.set_float_data([1.] * 3 * 224 * 224)
```

参数：

- `float_data(list)` - 待设置的float型数据

返回：`None`

返回类型：`None`



### `set_lod(lod)`

设置Tensor的LoD信息。

参数：

- `lod(list[list])` - Tensor的LoD信息

返回：`None`

返回类型：`None`



### `lod()`

获取Tensor的LoD信息

参数：

- `None`

返回：`Tensor`的LoD信息

返回类型：`list[list]`
