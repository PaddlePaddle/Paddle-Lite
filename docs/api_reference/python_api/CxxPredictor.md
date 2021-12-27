## CxxPredictor

```c++
class CxxPredictor
```

`CxxPredictor`是Paddle Lite的预测器，由`create_paddle_predictor`根据`CxxConfig`进行创建。用户可以根据CxxPredictor提供的接口设置输入数据、执行模型预测、获取输出以及获得当前使用lib的版本信息等。

示例：

```python
from paddlelite.lite import *
import numpy as np
import argparse

# Command arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_file", default="", type=str, help="Model file")
parser.add_argument(
    "--param_file", default="", type=str, help="Combined model param file")
parser.add_argument(
    "--model_dir", default="", type=str, help="Non-combined Model dir path")
args = parser.parse_args()

# 1. 设置CxxConfig
config = CxxConfig()
if args.model_file != '' and args.param_file != '':
    config.set_model_file(args.model_file)
    config.set_param_file(args.param_file)
else:
    config.set_model_dir(args.model_dir)
places = [Place(TargetType.X86, PrecisionType.FP32)]
config.set_valid_places(places)

# 2. 创建CxxPredictor
predictor = create_paddle_predictor(config)

# 3. 设置输入数据
input_tensor = predictor.get_input(0)
input_tensor.from_numpy(np.ones((1, 3, 224, 224)).astype("float32"))

# 4. 运行模型
predictor.run()

# 5. 获取输出数据
output_tensor = predictor.get_output(0)
output_data = output_tensor.numpy()
print(output_data)
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
