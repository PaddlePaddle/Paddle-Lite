# 模型量化-动态离线量化

本文首先简单介绍动态离线量化，然后说明产出量化模型，最后阐述量化模型预测。

## 1 简介

动态离线量化，将模型中特定OP的权重从FP32类型量化成INT8/16类型。

该量化模型有两种预测方式：
* 第一种是反量化预测方式，即是首先将INT8/16类型的权重反量化成FP32类型，然后再使用FP32浮运算运算进行预测；
* 第二种量化预测方式，即是预测中动态计算量化OP输入的量化信息，基于量化的输入和权重进行INT8整形运算。

注意：目前PaddleLite仅仅支持第一种反量化预测方式。

使用条件：
* 有训练好的预测模型

使用步骤：
* 产出量化模型：使用PaddlePaddle调用动态离线量化离线量化接口，产出量化模型
* 量化模型预测：使用PaddleLite加载量化模型进行预测推理

优点：
* 权重量化成INT16类型，模型精度不受影响，模型大小为原始的1/2
* 权重量化成INT8类型，模型精度会受到影响，模型大小为原始的1/4

缺点：
* 目前只支持反量化预测方式，主要可以减小模型大小，对特定加载权重费时的模型可以起到一定加速效果

## 2 产出量化模型

PaddleLite opt工具和PaddleSlim都提供了动态离线量化功能，两者原理相似，都可以产出动态离线量化的模型。

### 2.1 使用PaddleLite opt产出量化模型

PaddleLite opt工具将动态离线量化功能集成到模型转换中，使用简便，只需要设置对应参数，就可以产出优化后的量化模型。

#### 2.1.1 准备工具opt

参考[opt文档](./model_optimize_tool)，准备opt工具，其中可执行文件opt和python版本opt都提供了动态图离线量化功能。

#### 2.1.2 准备模型

准备已经训练好的FP32预测模型，即 `save_inference_model()` 保存的模型。

#### 2.1.3 产出量化模型

参考[opt文档](./model_optimize_tool)中使用opt工具的方法，在模型优化中启用动态离线量化方法产出优化后的量化模型。

如果是使用可执行文件opt工具，参考[直接下载并执行opt可执行工具](./opt/opt_bin)。
设置常规模型优化的参数后，可以通过 `--quant_model` 设置是否使用opt中的动态离线量化功能，通过 `--quant_type` 参数指定opt中动态离线量化功能的量化类型，可以设置为QUANT_INT8和QUANT_INT16，即分别量化为int8和int16。量化为int8对模型精度有一点影响，模型体积大概减小4倍。量化为int16对模型精度基本没有影响，模型体积大概减小2倍。
举例如下：
```shell
./opt \
    --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=naive_buffer \
    --optimize_out= <output_optimize_model_dir>\
    --quant_model=true \
    --quant_type=QUANT_INT16
```

如果使用python版本opt工具，请参考[安装 python版本opt后，使用终端命令](./opt/opt_python)和[安装python版本opt后，使用python脚本](../api_reference/python_api/opt)，都有介绍设置动态离线量化的参数和方法。

### 2.2 使用PaddleSlim产出量化模型

大家可以使用PaddleSlim调用动态离线量化接口，得到量化模型。

#### 2.2.1 安装PaddleSlim

参考PaddleSlim[文档](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)进行安装。

#### 2.2.2 准备模型

准备已经训练好的FP32预测模型，即 `save_inference_model()` 保存的模型。

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

动态离线量化api的详细介绍，请参考[链接](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/quantization_api.html#quant-post-dynamic)。

## 3 量化模型预测

目前，对于动态离线量化产出的量化模型，只能使用PaddleLite进行预测部署。

很简单，首先使用PaddleLite提供的模型转换工具（opt）将量化模型转换成移动端预测的模型，然后加载转换后的模型进行预测部署。

注意，PaddleLite 2.3版本才支持动态离线量化产出的量化，所以转换工具和预测库必须是2.3及之后的版本。

### 3.1 模型转换

参考[模型转换](../user_guides/model_optimize_tool)准备模型转换工具，建议从Release页面下载。

参考[模型转换](../user_guides/model_optimize_tool)使用模型转换工具。
比如在安卓手机ARM端进行预测，模型转换的命令为：
```bash
./opt --model_dir=./mobilenet_v1_quant \
      --optimize_out_type=naive_buffer \
      --optimize_out=mobilenet_v1_quant_opt \
      --valid_targets=arm
```

### 3.2 量化模型预测

和FP32模型一样，转换后的量化模型可以在Android/IOS APP中加载预测，建议参考[C++ Demo](../quick_start/cpp_demo)、[Java Demo](../quick_start/java_demo)、[Android/IOS Demo](../demo_guides/android_app_demo)。
