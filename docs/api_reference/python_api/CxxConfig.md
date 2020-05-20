## CxxConfig

```python
class CxxConfig;
```

`CxxConfig`用来配置构建CxxPredictor的配置信息，如protobuf格式的模型地址、能耗模式、工作线程数、place信息等等。

示例：

```python
from paddlelite.lite import *

config = CxxConfig()
# 设置模型目录，加载非combined模型时使用
config.set_model_dir(<your_model_dir_path>)
# 设置工作线程数(该接口只支持armlinux)
# config.set_threads(4);
# 设置能耗模式(该接口只支持armlinux)
# config.set_power_mode(PowerMode.LITE_POWER_NO_BIND)
# 设置valid places
places = [Place(TargetType.ARM, PrecisionType.FP32)]
config.set_valid_places(places)

# 根据CxxConfig创建CxxPredictor
predictor = lite.create_paddle_predictor(config)
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
from paddlelite.lite import *

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

设置CPU能耗模式，该接口只支持`armlinux`平台。若不设置，则默认使用`PowerMode.LITE_POWER_HIGH`。

*注意：只在开启`OpenMP`时生效，否则系统自动调度。此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `mode(PowerMode)` - CPU能耗模式

返回：`None`

返回类型：`None`



### `power_mode()`

获取设置的CPU能耗模式，该接口只支持`armlinux`平台。

*注意：此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `None`

返回：设置的CPU能耗模式

返回类型：`PowerMode`



### `set_threads(threads)`

设置工作线程数，该接口只支持`armlinux`平台。若不设置，则默认使用单线程。

*注意：只在开启`OpenMP`的模式下生效，否则只使用单线程。此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `threads(int)` - 工作线程数

返回：`None`

返回类型：`None`



### `threads()`

获取设置的工作线程数，该接口只支持`armlinux`平台。

*注意：此函数只在使用`LITE_WITH_ARM`编译选项下生效。*

参数：

- `None`

返回：工作线程数

返回类型：`int`
