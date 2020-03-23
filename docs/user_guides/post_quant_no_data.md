# 模型量化-无校准数据训练后量化

本文首先简单介绍无校准数据训练后量化，然后说明产出量化模型，最好阐述量化模型预测。

## 1 简介

无校准数据训练后量化，将模型中特定OP的权重从FP32类型量化成INT8/16类型，可以减小预测模型的大小。使用该量化模型预测，首先将INT8/16类型的权重反量化成FP32类型，然后再进行预测。

使用条件：
* 有训练好的预测模型

使用步骤：
* 产出量化模型：使用PaddlePaddle调用无校准数据训练后量化接口，产出量化模型
* 量化模型预测：使用PaddleLite加载量化模型进行预测推理

优点：
* 权重量化成INT16类型，模型精度不受影响，模型大小为原始的1/2
* 权重量化成INT8类型，模型精度会受到影响，模型大小为原始的1/4

缺点：
* 暂无

## 2 产出量化模型

大家可以使用PaddlePaddle调用无校准数据训练后量化接口，得到量化模型。

### 2.1 安装PaddlePaddle

参考PaddlePaddle[官网](https://www.paddlepaddle.org.cn/install/quick)，安装PaddlePaddle CPU/GPU 1.7版本。

### 2.2 准备模型

准备已经训练好的FP32预测模型，即 `save_inference_model()` 保存的模型。

### 2.3 调用无校准数据训练后量化

对于调用无校准数据训练后量化，首先给出一个例子。

```python
from paddle.fluid.contrib.slim.quantization import WeightQuantization

model_dir = path/to/fp32_model_params
save_model_dir = path/to/save_model_path
weight_quant = WeightQuantization(model_dir=model_dir)
weight_quant.quantize_weight_to_int(save_model_dir=save_model_dir,
                                    weight_bits=16,
                                    quantizable_op_type=['conv2d', 'depthwise_conv2d', 'mul'])
```

对于调用无校准数据训练后量化，以下对api接口进行详细介绍。

```python
class WeightQuantization(model_dir, model_filename=None, params_filename=None)
```
参数说明如下：
* model_dir(str)：待量化模型的路径，其中保存模型文件和权重文件。
* model_filename(str, optional)：待量化模型的模型文件名，如果模型文件名不是`__model__`，则需要使用model_filename设置模型文件名。
* params_filename(str, optional)：待量化模型的权重文件名，如果所有权重保存成一个文件，则需要使用params_filename设置权重文件名。

```python
WeightQuantization.quantize_weight_to_int(save_model_dir,
                                          save_model_filename=None,
                                          save_params_filename=None,
                                          quantizable_op_type=['conv2d', 'mul'],
                                          weight_bits=8,
                                          threshold_rate=0.0)
```
参数说明如下：
* save_model_dir(str)：保存量化模型的路径。
* save_model_filename(str, optional)：如果save_model_filename等于None，则模型的网络结构保存到__model__文件，如果save_model_filename不等于None，则模型的网络结构保存到特定的文件。默认为None。
* save_params_filename(str, optional)：如果save_params_filename等于None，则模型的参数分别保存到一系列文件中，如果save_params_filename不等于None，则模型的参数会保存到一个文件中，文件名为设置的save_params_filename。默认为None。
* quantizable_op_type(list[str]): 需要量化的op类型，默认是`['conv2d', 'mul']`，列表中的值可以是任意支持量化的op类型 `['conv2d', 'depthwise_conv2d', 'mul']`。
* weight_bits(int, optional)：权重量化保存的比特数，可以是8~16，一般设置为8/16。默认为8。


## 3 量化模型预测

目前，对于无校准数据训练后量化产出的量化模型，不支持PaddlePaddle加载执行，只能使用PaddleLite进行预测部署。

很简单，首先使用PaddleLite提供的模型转换工具（opt）将量化模型转换成移动端预测的模型，然后加载转换后的模型进行预测部署。

注意，PaddleLite 2.3版本才支持无校准数据训练后量化产出的量化，所以转换工具和预测库必须是2.3及之后的版本。

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

和FP32模型一样，转换后的量化模型可以在Android/IOS APP中加载预测，建议参考[C++ Demo](../demo_guides/cpp_demo)、[Java Demo](../demo_guides/java_demo)、[Android/IOS Demo](../demo_guides/android_app_demo)。
