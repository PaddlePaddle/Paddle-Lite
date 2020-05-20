## MobileConfig

```python
class MobileConfig;
```

`MobileConfig`用来配置构建LightPredictor的配置信息，如NaiveBuffer格式的模型地址、能耗模式、工作线程数等等。

示例：

```python
from paddlelite.lite import *

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
