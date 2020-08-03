## Tensor

```c++
class Tensor
```

Tensor是Paddle-Lite的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置Shape、数据、LoD信息等。

*注意：用户应使用`CxxPredictor`或`LightPredictor`的`get_input`和`get_output`接口获取输入/输出的`Tensor`。*

示例：

```python
from paddlelite.lite import *

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
