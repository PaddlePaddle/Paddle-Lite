# 离线量化-动态离线量化

本部分首先简单介绍动态离线量化，然后说明产出量化模型，最后阐述量化模型预测。

## 1 简介

动态离线量化，将模型中特定 OP 的权重从 FP32 类型量化成 INT8/16 类型。

该量化模型有两种预测方式：
* 第一种是反量化预测方式，即是首先将 INT8/16 类型的权重反量化成 FP32 类型，然后再使用 FP32 浮运算运算进行预测；
* 第二种量化预测方式，即是预测中动态计算量化 OP 输入的量化信息，基于量化的输入和权重进行 INT8 整形运算。

注意：目前 Paddle Lite 仅支持第一种反量化预测方式。

使用条件：
* 有训练好的预测模型

使用步骤：
* 产出量化模型：使用 PaddlePaddle 调用动态离线量化离线量化接口，产出量化模型
* 量化模型预测：使用 Paddle Lite 加载量化模型进行预测推理

优点：
* 权重量化成 INT16 类型，模型精度不受影响，模型大小为原始的1/2
* 权重量化成 INT8 类型，模型精度会受到影响，模型大小为原始的1/4

缺点：
* 目前只支持反量化预测方式，主要可以减小模型大小，对特定加载权重费时的模型可以起到一定加速效果

和静态离线量化相比，目前 Paddle Lite 只支持反量化预测方式，对于预测过程的加速有限；但是动态离线量化不需要训练数据即可完成量化，达到减小模型大小的目的。

## 2 产出量化模型

Paddle Lite OPT 工具和 PaddleSlim 都提供了动态离线量化功能，两者原理相似，都可以产出动态离线量化的模型。

### 2.1 使用 Paddle Lite OPT 产出量化模型

Paddle Lite OPT 工具将动态离线量化功能集成到模型转换中，使用简便，只需要设置对应参数，就可以产出优化后的量化模型。

#### 2.1.1 准备工具 OPT

参考[ OPT 文档](../model_optimize_tool)，准备 OPT 工具，其中可执行文件 opt 和 python 版本 opt 都提供了动态图离线量化功能。

#### 2.1.2 准备模型

准备已经训练好的 FP32 预测模型，即 `save_inference_model()` 保存的模型。

#### 2.1.3 产出量化模型

参考[ OPT 文档](../model_optimize_tool)中使用 OPT 工具的方法，在模型优化中启用动态离线量化方法产出优化后的量化模型。

如果是使用可执行文件 OPT 工具，参考[直接下载并执行 OPT 可执行工具](../opt/opt_bin)。
设置常规模型优化的参数后，可以通过 `--quant_model` 设置是否使用 OPT 中的动态离线量化功能，通过 `--quant_type` 参数指定 OPT 中动态离线量化功能的量化类型，可以设置为 QUANT_INT8 和 QUANT_INT16 ，即分别量化为 int8 和 int16 。量化为 int8 对模型精度有一点影响，模型体积大概减小4倍。量化为 int16 对模型精度基本没有影响，模型体积大概减小2倍。
举例如下：
```shell
./OPT \
    --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=naive_buffer \
    --optimize_out= <output_optimize_model_dir>\
    --quant_model=true \
    --quant_type=QUANT_INT16
```

如果使用 python 版本 OPT 工具，请参考[安装 python 版本 OPT 后，使用终端命令](../opt/opt_python)和[安装 python 版本 OPT 后，使用 python 脚本](../../api_reference/python_api/OPT)，都有介绍设置动态离线量化的参数和方法。

### 2.2 使用 PaddleSlim 产出量化模型

大家可以使用 PaddleSlim 调用动态离线量化接口，得到量化模型。

#### 2.2.1 安装 PaddleSlim

参考 PaddleSlim [文档](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)进行安装。

#### 2.2.2 准备模型

准备已经训练好的 FP32 预测模型，即 `save_inference_model()` 保存的模型。

#### 2.2.3 调用动态离线量化

对于调用动态离线量化，首先给出一个例子。

```python
from paddleslim.quant import quant_post_dynamic

model_dir = path/to/fp32_model_params
save_model_dir = path/to/save_model_path
quant_post_dynamic(model_dir=model_dir,
                   save_model_dir=save_model_dir,
                   weight_bits=8,
                   quantizable_op_type=['conv2d', 'mul'],
                   weight_quantize_type="channel_wise_abs_max",
                   generate_test_model=False)
```

执行完成后，可以在 `save_model_dir/quantized_model` 目录下得到量化模型。

## 3 量化模型预测

目前，对于动态离线量化产出的量化模型，只能使用 Paddle Lite 进行预测部署。

很简单，首先使用 Paddle Lite 提供的模型转换工具（OPT）将量化模型转换成移动端预测的模型，然后加载转换后的模型进行预测部署。

注意，Paddle Lite 2.3 版本才支持动态离线量化产出的量化，所以转换工具和预测库必须是2.3及之后的版本。

### 3.1 模型转换

参考[模型转换](../model_optimize_tool)准备模型转换工具，建议从 Release 页面下载。

参考[模型转换](../model_optimize_tool)使用模型转换工具。
比如在安卓手机 ARM 端进行预测，模型转换的命令为：
```bash
./OPT --model_dir=./mobilenet_v1_quant \
      --optimize_out_type=naive_buffer \
      --optimize_out=mobilenet_v1_quant_opt \
      --valid_targets=arm
```

### 3.2 量化模型预测

和 FP32 模型一样，转换后的量化模型可以在 Android/IOS APP 中加载预测，建议参考 [C++ Demo](../cpp_demo)、[Java Demo](../java_demo)、[Android/IOS Demo](../../demo_guides/android_app_demo)。
