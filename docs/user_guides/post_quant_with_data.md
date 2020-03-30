# 模型量化-有校准数据训练后量化

本文首先简单介绍有校准数据训练后量化，然后说明产出量化模型、量化模型预测，最后给出一个使用示例。
如果想快速上手，大家可以先参考使用示例，再查看详细使用方法。

## 1 简介

有校准数据训练后量化，使用少量校准数据计算量化因子，可以快速得到量化模型。使用该量化模型进行预测，可以减少计算量、降低计算内存、减小模型大小。

有校准数据训练后量化中，有两种计算量化因子的方法，非饱和量化方法和饱和量化方法。非饱和量化方法计算整个Tensor的绝对值最大值`abs_max`，将其映射为127。饱和量化方法使用KL散度计算一个合适的阈值`T` (`0<T<mab_max`)，将其映射为127。一般而言，待量化Op的权重采用非饱和量化方法，待量化Op的激活（输入和输出）采用饱和量化方法 。

使用条件：
* 有训练好的预测模型
* 有少量校准数据，比如100~500张图片

使用步骤：
* 产出量化模型：使用PaddlePaddle或者PaddleSlim调用有校准数据训练后量化接口，产出量化模型
* 量化模型预测：使用PaddleLite加载量化模型进行预测推理

优点：
* 减小计算量、降低计算内存、减小模型大小
* 不需要大量训练数据
* 快速产出量化模型，简单易用

缺点：
* 对少部分的模型，尤其是计算量小、精简的模型，量化后精度可能会受到影响

## 2 产出量化模型

大家可以使用PaddlePaddle或者PaddleSlim调用有校准数据训练后量化接口，得到量化模型。本文主要介绍使用PaddlePaddle产出量化模型，使用PaddleSlim可以参考[文档](https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim)。

### 2.1 安装PaddlePaddle

参考PaddlePaddle[官网](https://www.paddlepaddle.org.cn/install/quick)，安装PaddlePaddle CPU/GPU 1.7版本。

### 2.2 准备模型和校准数据

准备已经训练好的FP32预测模型，即 `save_inference_model()` 保存的模型。
准备校准数据集，校准数据集应该是测试集/训练集中随机挑选的一部分，量化因子才会更加准确。对常见的视觉模型，建议校准数据的数量为100~500张图片。

### 2.3 配置校准数据生成器

有校准数据训练后量化内部使用异步数据读取的方式读取校准数据，大家只需要根据模型的输入，配置读取数据的sample_generator。sample_generator是Python生成器，**必须每次返回单个样本数据**，会用作`DataLoader.set_sample_generator()`的数据源。
建议参考[异步数据读取文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/data_preparing/use_py_reader.html)和本文示例，学习如何配置校准数据生成器。

### 2.4 调用有校准数据训练后量化

对于调用有校准数据训练后量化，首先给出一个例子，让大家有个直观了解。

```python
import paddle.fluid as fluid
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization

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
ptq = PostTrainingQuantization(
            executor=exe,
            sample_generator=sample_generator,
            model_dir=model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
            batch_size=batch_size,
            batch_nums=batch_nums,
            algo=algo,
            quantizable_op_type=quantizable_op_type)
ptq.quantize()
ptq.save_quantized_model(save_model_path)
```

对于调用有校准数据训练后量化，以下对接口进行详细介绍。

``` python
class PostTrainingQuantization(
                 executor=None,
                 scope=None,
                 model_dir=None,
                 model_filename=None,
                 params_filename=None,
                 sample_generator=None,
                 batch_size=10,
                 batch_nums=None,
                 algo="KL",
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 is_full_quantize=False,
                 weight_bits=8,
                 activation_bits=8,
                 is_use_cache_file=False,
                 cache_dir="./temp_post_training"):
```
调用上述api，传入必要的参数。参数说明如下：
* executor(fluid.Executor)：执行模型的executor，可以指定在cpu或者gpu上执行。
* scope(fluid.Scope, optional)：模型运行时使用的scope，默认为None，则会使用global_scope()。行首有optional，说明用户可以不设置该输入参数，直接使用默认值，下同。
* model_dir(str)：待量化模型的路径，其中保存模型文件和权重文件。
* model_filename(str, optional)：待量化模型的模型文件名，如果模型文件名不是`__model__`，则需要使用model_filename设置模型文件名。
* params_filename(str, optional)：待量化模型的权重文件名，如果所有权重保存成一个文件，则需要使用params_filename设置权重文件名。
* sample_generator(Python Generator)：配置的校准数据生成器。
* batch_size(int, optional)：一次读取校准数据的数量。
* batch_nums(int, optional)：读取校准数据的次数。如果设置为None，则从sample_generator中读取所有校准数据进行训练后量化；如果设置为非None，则从sample_generator中读取`batch_size*batch_nums`个校准数据。
* algo(str, optional)：计算待量化激活Tensor的量化因子的方法。设置为`KL`，则使用饱和量化方法，设置为`direct`，则使用非饱和量化方法。默认为`KL`。
* quantizable_op_type(list[str], optional): 需要量化的op类型，默认是`["conv2d", "depthwise_conv2d", "mul"]`，列表中的值可以是任意支持量化的op类型。
* is_full_quantize(bool, optional)：是否进行全量化。设置为True，则对模型中所有支持量化的op进行量化；设置为False，则只对`quantizable_op_type` 中op类型进行量化。目前支持的量化类型如下：'conv2d', 'depthwise_conv2d', 'mul', "pool2d", "elementwise_add", "concat", "softmax", "argmax", "transpose", "equal", "gather", "greater_equal", "greater_than", "less_equal", "less_than", "mean", "not_equal", "reshape", "reshape2", "bilinear_interp", "nearest_interp", "trilinear_interp", "slice", "squeeze", "elementwise_sub"。
* weight_bits(int, optional)：权重量化的比特数，可以设置为1~16。PaddleLite目前仅支持加载权重量化为8bit的量化模型。
* activation_bits(int, optional)： 激活量化的比特数，可以设置为1~16。PaddleLite目前仅支持加载激活量化为8bit的量化模型。
* is_use_cache_file(bool, optional)：是否使用缓存文件。如果设置为True，训练后量化过程中的采样数据会保存到磁盘文件中；如果设置为False，所有采样数据会保存到内存中。当待量化的模型很大或者校准数据数量很大，建议设置is_use_cache_file为True。默认为False。
* cache_dir(str, optional)：当is_use_cache_file等于True，会将采样数据保存到该文件中。量化完成后，该文件中的临时文件会自动删除。

```python
PostTrainingQuantization.quantize()
```
调用上述接口开始训练后量化。根据校准数据数量、模型的大小和量化op类型不同，训练后量化需要的时间也不一样。比如使用ImageNet2012数据集中100图片对`MobileNetV1`进行训练后量化，花费大概1分钟。

```python
PostTrainingQuantization.save_quantized_model(save_model_path)
```
调用上述接口保存训练后量化模型，其中save_model_path为保存的路径。

训练后量化支持部分量化功能：
* 方法1：设置quantizable_op_type，则只会对quantizable_op_type中的Op类型进行量化，模型中其他Op类型保持不量化。
* 方法2：构建网络的时候，将不需要量化的特定Op定义在 `skip_quant` 的name_scope中，则可以跳过特定Op的量化，示例如下。
```python
with fluid.name_scope('skip_quant'):
    pool = fluid.layers.pool2d(input=hidden, pool_size=2, pool_type='avg', pool_stride=2)
    # 不对pool2d进行量化
```

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

## 4 使用示例

### 4.1 产出量化模型

参考本文 “2.1 安装PaddlePaddle” 安装PaddlePaddle。

下载[打包文件](https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/quantization_demo/post_training_quantization_withdata.tgz)，解压到本地。
```bash
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/quantization_demo/post_training_quantization_withdata.tgz
tar zxvf post_training_quantization_withdata.tgz
cd post_training_quantization_withdata
```

执行下面的命令，自动下载预测模型(mobilenetv1_fp32_model)和校准数据集，然后调用有校准数据训练后方法产出量化模型。
```bash
sh run_post_training_quanzation.sh
```

量化模型保存在mobilenetv1_int8_model文件夹中。

### 4.2 量化模型预测

下载测试文件（[benchmark_bin](https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/quantization_demo/benchmark_bin)）或者参考[Benchmark测试方法](../benchmark/benchmark_tools)编译测试文件。

将mobilenetv1_fp32_model、mobilenetv1_int8_model和benchmark_bin文件都保存到手机上。
```bash
adb push mobilenetv1_fp32_model /data/local/tmp
adb push mobilenetv1_int8_model /data/local/tmp
chmod 777 benchmark_bin
adb push benchmark_bin /data/local/tmp
```

测试量化模型和原始模型的性能，依次执行下面命令：
```bash
./benchmark_bin --is_quantized_model=true --run_model_optimize=true  --result_filename=res.txt --warmup=10 --repeats=30  --model_dir=mobilenetv1_int8_model/
./benchmark_bin --is_quantized_model=true --run_model_optimize=true  --result_filename=res.txt --warmup=10 --repeats=30 --model_dir=mobilenetv1_fp32_model/
cat res.txt
```

在res.txt文件中可以看到INT8量化模型和FP32原始模型的速度。
举例来说，在骁龙855手机、单线程的情况下测试mobilenetv1，INT8量化模型的计算时间是14.52ms，FP32原始模型的计算时间是31.7ms。
