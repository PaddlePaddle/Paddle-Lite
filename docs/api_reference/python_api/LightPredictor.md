## LightPredictor

```c++
class LightPredictor
```

`LightPredictor`是Paddle Lite的预测器，由`create_paddle_predictor`根据`MobileConfig`进行创建。用户可以根据LightPredictor提供的接口设置输入数据、执行模型预测、获取输出以及获得当前使用lib的版本信息等。

示例：

```python
from __future__ import print_function
from paddlelite.lite import *
import numpy as np
import argparse

# Command arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_file", type=str, help="the path to optimized model after opt tool")
args = parser.parse_args()

# 1. 设置MobileConfig
config = MobileConfig()
config.set_model_from_file(args.model_file)

# 2. 创建LightPredictor
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
