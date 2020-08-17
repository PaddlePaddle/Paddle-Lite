# 模型量化-静态离线量化

## 1 简介

静态离线量化，使用少量校准数据计算量化因子，可以快速得到量化模型。使用该量化模型进行预测，可以减少计算量、降低计算内存、减小模型大小。

静态离线量化中，有两种计算量化因子的方法，非饱和量化方法和饱和量化方法。非饱和量化方法计算整个Tensor的绝对值最大值`abs_max`，将其映射为127。饱和量化方法使用KL散度计算一个合适的阈值`T` (`0<T<mab_max`)，将其映射为127。一般而言，待量化Op的权重采用非饱和量化方法，待量化Op的激活（输入和输出）采用饱和量化方法 。

使用条件：
* 有训练好的预测模型
* 有少量校准数据，比如100~500张图片

使用步骤：
* 产出量化模型：使用PaddleSlim调用静态离线量化接口，产出量化模型
* 量化模型预测：使用PaddleLite加载量化模型进行预测推理

优点：
* 减小计算量、降低计算内存、减小模型大小
* 不需要大量训练数据
* 快速产出量化模型，简单易用

缺点：
* 对少部分的模型，尤其是计算量小、精简的模型，量化后精度可能会受到影响

## 2 产出量化模型

大家可以使用PaddleSlim调用静态离线量化接口，得到量化模型。

### 2.1 安装PaddleSlim

参考PaddleSlim[文档](https://paddlepaddle.github.io/PaddleSlim/install.html)进行安装。

### 2.2 准备模型和校准数据

准备已经训练好的FP32预测模型，即 `save_inference_model()` 保存的模型。
准备校准数据集，校准数据集应该是测试集/训练集中随机挑选的一部分，量化因子才会更加准确。对常见的视觉模型，建议校准数据的数量为100~500张图片。

### 2.3 配置校准数据生成器

静态离线量化内部使用异步数据读取的方式读取校准数据，大家只需要根据模型的输入，配置读取数据的sample_generator。sample_generator是Python生成器，**必须每次返回单个样本数据**，会用作`DataLoader.set_sample_generator()`的数据源。
建议参考[异步数据读取文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/data_preparing/static_mode/use_py_reader.html)和本文示例，学习如何配置校准数据生成器。

### 2.4 调用静态离线量化

对于调用静态离线量化，首先给出一个例子，让大家有个直观了解。

```python
import paddle.fluid as fluid
from paddleslim.quant import quant_post

exe = fluid.Executor(fluid.CPUPlace())
model_dir = path/to/fp32_model_params
# set model_filename as None when the filename is __model__, 
# otherwise set it as the real filename
model_filename = None 
# set params_filename as None when all parameters were saved in 
# separate files, otherwise set it as the real filename
params_filename = None
save_model_path = path/to/save_model_path
# prepare the sample generator according to the model, and the 
# sample generator must return a sample every time. The reference
# document: https://www.paddlepaddle.org.cn/documentation/docs/zh
# /user_guides/howto/prepare_data/use_py_reader.html
sample_generator = your_sample_generator
batch_size = 10
batch_nums = 10
algo = "KL"
quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
quant_post(executor=exe,
           model_dir=model_dir,
           model_filename=model_filename,
           params_filename=params_filename,
           quantize_model_path=save_model_path,
           sample_generator=sample_generator,
           batch_size=batch_size,
           batch_nums=batch_nums,
           algo=algo,
           quantizable_op_type=quantizable_op_type)
```

快速开始请参考[文档](https://paddlepaddle.github.io/PaddleSlim/quick_start/quant_post_tutorial.html#)。

API接口请参考[文档](https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-post)。

Demo请参考[文档](https://github.com/PaddlePaddle/PaddleSlim/tree/release/1.0.1/demo/quant/quant_post)。

## 3 量化模型预测

首先，使用PaddleLite提供的模型转换工具（model_optimize_tool）将量化模型转换成移动端预测的模型，然后加载转换后的模型进行预测部署。

### 3.1 模型转换

参考[模型转换](../user_guides/model_optimize_tool)准备模型转换工具，建议从Release页面下载。

参考[模型转换](../user_guides/model_optimize_tool)使用模型转换工具，参数按照实际情况设置。比如在安卓手机ARM端进行预测，模型转换的命令为：
```bash
./opt --model_dir=./mobilenet_v1_quant \
      --optimize_out_type=naive_buffer \
      --optimize_out=mobilenet_v1_quant_opt \
      --valid_targets=arm
```

### 3.2 量化模型预测

和FP32模型一样，转换后的量化模型可以在Android/IOS APP中加载预测，建议参考[C++ Demo](../demo_guides/cpp_demo)、[Java Demo](../demo_guides/java_demo)、[Android/IOS Demo](../demo_guides/android_app_demo)。
