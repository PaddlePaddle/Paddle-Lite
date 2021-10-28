# 模型量化

首先我们介绍一下 Paddle 支持的模型量化方法，让大家有一个整体的认识。

Paddle 模型量化包含三种量化方法，分别是动态离线量化方法、静态离线量化方法和量化训练方法。

下图展示了如何选择模型量化方法。

![img](https://user-images.githubusercontent.com/52520497/95644539-e7f23500-0ae9-11eb-80a8-596cfb285e17.png)

下图综合对比了模型量化方法的使用条件、易用性、精度损失和预期收益。

![image](https://user-images.githubusercontent.com/52520497/118938675-7b91bb00-b981-11eb-9666-706d3828f216.png)


大家可以根据不同情况选用不同的量化方法，有几点着重注意：
* 动态离线量化方法主要用于减小模型体积
* 静态离线量化方法和量化训练方法既可以减小模型体积，也可以加快性能，性能加速基本相同
* 静态离线量化方法比量化训练方法更加简单，一般建议首先使用静态离线量化方法，如果量化模型精度损失较大，再尝试使用量化训练方法。

本章节主要介绍使用 PaddleSlim 量化训练方法产出的量化模型，使用 Paddle Lite 加载量化模型进行推理部署，

## 量化训练

### 1 简介

量化训练是使用较多训练数据，对训练好的预测模型进行量化。该方法使用模拟量化的思想，在训练阶段更新权重，实现减小量化误差。

使用条件：
* 有预训练模型
* 有较多训练数据（通常大于5000）

使用步骤：
* 产出量化模型：使用 PaddlePaddle 调用量化训练接口，产出量化模型
* 量化模型预测：使用 Paddle Lite 加载量化模型进行预测推理

优点：
* 减小计算量、降低计算内存、减小模型大小
* 模型精度受量化影响小

缺点：
* 使用条件较苛刻，使用门槛稍高

建议首先使用“静态离线量化”方法对模型进行量化，然后使用使用量化模型进行预测。如果该量化模型的精度达不到要求，再使用“量化训练”方法。

### 2 产出量化模型

目前，PaddleSlim 的量化训练主要针对卷积层和全连接层，对应算子是 conv2d、depthwise_conv2d、conv2d_tranpose 和 mul。Paddle Lite 支持运行 PaddleSlim 量化训练产出的模型，可以进一步加快模型在移动端的执行速度。

温馨提示：如果您是初次接触 PaddlePaddle 框架，建议首先学习[使用文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/index_cn.html)。

使用 PaddleSlim 模型压缩工具训练量化模型，请参考文档：
* 量化训练[快速开始教程](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/index.html)
* 量化训练[API接口说明](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/index.html)
* 量化训练[Demo](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant)


### 3 使用 Paddle Lite 运行量化模型推理

首先，使用 Paddle Lite 提供的模型转换工具（model_optimize_tool）将量化模型转换成移动端预测的模型，然后加载转换后的模型进行预测部署。

#### 3.1 模型转换

参考[模型转换](../user_guides/model_optimize_tool)准备模型转换工具，建议从 Release 页面下载。

参考[模型转换](../user_guides/model_optimize_tool)使用模型转换工具，参数按照实际情况设置。比如在安卓手机ARM端进行预测，模型转换的命令为：
```bash
./OPT --model_dir=./mobilenet_v1_quant \
      --optimize_out_type=naive_buffer \
      --optimize_out=mobilenet_v1_quant_opt \
      --valid_targets=arm
```

#### 3.2 量化模型预测

和 FP32 模型一样，转换后的量化模型可以在 Android/IOS APP 中加载预测，建议参考[C++ Demo](./cpp_demo)、[Java Demo](./java_demo)、[Android/IOS Demo](../demo_guides/android_app_demo)。


### FAQ

**问题**：Compiled with WITH_GPU, but no GPU found in runtime

**解答**：检查本机是否支持 GPU 训练，如果不支持请使用 CPU 训练。如果在 docker 进行 GPU 训练，请使用 nvidia_docker 启动容器。

**问题**：Inufficient GPU memory to allocation. at [/paddle/paddle/fluid/platform/gpu_info.cc:262]
  
**解答**：正确设置 run.sh 脚本中`CUDA_VISIBLE_DEVICES`，确保显卡剩余内存大于需要内存。


## 离线量化-动态离线量化

本部分首先简单介绍动态离线量化，然后说明产出量化模型，最后阐述量化模型预测。

### 1 简介

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

### 2 产出量化模型

Paddle Lite OPT 工具和 PaddleSlim 都提供了动态离线量化功能，两者原理相似，都可以产出动态离线量化的模型。

#### 2.1 使用 Paddle Lite OPT 产出量化模型

Paddle Lite OPT 工具将动态离线量化功能集成到模型转换中，使用简便，只需要设置对应参数，就可以产出优化后的量化模型。

##### 2.1.1 准备工具 OPT

参考[ OPT 文档](./model_optimize_tool)，准备 OPT 工具，其中可执行文件 opt 和 python 版本 opt 都提供了动态图离线量化功能。

##### 2.1.2 准备模型

准备已经训练好的 FP32 预测模型，即 `save_inference_model()` 保存的模型。

##### 2.1.3 产出量化模型

参考[ OPT 文档](./model_optimize_tool)中使用 OPT 工具的方法，在模型优化中启用动态离线量化方法产出优化后的量化模型。

如果是使用可执行文件 OPT 工具，参考[直接下载并执行 OPT 可执行工具](./OPT/opt_bin)。
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

如果使用 python 版本 OPT 工具，请参考[安装 python 版本 OPT 后，使用终端命令](./OPT/opt_python)和[安装 python 版本 OPT 后，使用 python 脚本](../api_reference/python_api/OPT)，都有介绍设置动态离线量化的参数和方法。

#### 2.2 使用 PaddleSlim 产出量化模型

大家可以使用 PaddleSlim 调用动态离线量化接口，得到量化模型。

##### 2.2.1 安装 PaddleSlim

参考 PaddleSlim [文档](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)进行安装。

#### 2.2.2 准备模型

准备已经训练好的 FP32 预测模型，即 `save_inference_model()` 保存的模型。

##### 2.2.3 调用动态离线量化

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

### 3 量化模型预测

目前，对于动态离线量化产出的量化模型，只能使用 Paddle Lite 进行预测部署。

很简单，首先使用 Paddle Lite 提供的模型转换工具（OPT）将量化模型转换成移动端预测的模型，然后加载转换后的模型进行预测部署。

注意，Paddle Lite 2.3 版本才支持动态离线量化产出的量化，所以转换工具和预测库必须是2.3及之后的版本。

#### 3.1 模型转换

参考[模型转换](../user_guides/model_optimize_tool)准备模型转换工具，建议从 Release 页面下载。

参考[模型转换](../user_guides/model_optimize_tool)使用模型转换工具。
比如在安卓手机 ARM 端进行预测，模型转换的命令为：
```bash
./OPT --model_dir=./mobilenet_v1_quant \
      --optimize_out_type=naive_buffer \
      --optimize_out=mobilenet_v1_quant_opt \
      --valid_targets=arm
```

#### 3.2 量化模型预测

和 FP32 模型一样，转换后的量化模型可以在 Android/IOS APP 中加载预测，建议参考 [C++ Demo](./cpp_demo)、[Java Demo](./java_demo)、[Android/IOS Demo](../demo_guides/android_app_demo)。


## 离线量化-静态离线量化

#### 1 简介

静态离线量化，使用少量校准数据计算量化因子，可以快速得到量化模型。使用该量化模型进行预测，可以减少计算量、降低计算内存、减小模型大小。

静态离线量化中，有两种计算量化因子的方法，非饱和量化方法和饱和量化方法。非饱和量化方法计算整个 Tensor 的绝对值最大值`abs_max`，将其映射为127。饱和量化方法使用 KL 散度计算一个合适的阈值`T` (`0<T<mab_max`)，将其映射为127。一般而言，待量化 Op 的权重采用非饱和量化方法，待量化 Op 的激活（输入和输出）采用饱和量化方法 。

使用条件：
* 有训练好的预测模型
* 有少量校准数据，比如100~500张图片

使用步骤：
* 产出量化模型：使用 PaddleSlim 调用静态离线量化接口，产出量化模型
* 量化模型预测：使用 Paddle Lite 加载量化模型进行预测推理

优点：
* 减小计算量、降低计算内存、减小模型大小
* 不需要大量训练数据
* 快速产出量化模型，简单易用

缺点：
* 对少部分的模型，尤其是计算量小、精简的模型，量化后精度可能会受到影响

#### 2 产出量化模型

大家可以使用 PaddleSlim 调用静态离线量化接口，得到量化模型。

###### 2.1 安装 PaddleSlim

参考 PaddleSlim [文档](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)进行安装。

###### 2.2 准备模型和校准数据

准备已经训练好的 FP32 预测模型，即 `save_inference_model()` 保存的模型。
准备校准数据集，校准数据集应该是测试集/训练集中随机挑选的一部分，量化因子才会更加准确。对常见的视觉模型，建议校准数据的数量为100~500张图片。

###### 2.3 配置校准数据生成器

静态离线量化内部使用异步数据读取的方式读取校准数据，大家只需要根据模型的输入，配置读取数据的 sample_generator。sample_generator 是 Python 生成器，**必须每次返回单个样本数据**，会用作`DataLoader.set_sample_generator()`的数据源。
建议参考[异步数据读取文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/howto/prepare_data/use_py_reader.html)和本文示例，学习如何配置校准数据生成器。

###### 2.4 调用静态离线量化

对于调用静态离线量化，首先给出一个例子，让大家有个直观了解。

```python
import paddle.fluid as fluid
from paddleslim.quant import quant_post_static

exe = fluid.Executor(fluid.CPUPlace())
model_dir = path/to/fp32_model_params
## set model_filename as None when the filename is __model__, 
## otherwise set it as the real filename
model_filename = None 
## set params_filename as None when all parameters were saved in 
## separate files, otherwise set it as the real filename
params_filename = None
save_model_path = path/to/save_model_path
## prepare the sample generator according to the model, and the 
## sample generator must return a sample every time. The reference
## document: https://www.paddlepaddle.org.cn/documentation/docs/zh
## /user_guides/howto/prepare_data/use_py_reader.html
sample_generator = your_sample_generator
batch_size = 10
batch_nums = 10
algo = "KL"
quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
quant_post_static(executor=exe,
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
API 接口请参考[文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/index.html)。

Demo 请参考[文档](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_post)。

#### 3 量化模型预测

首先，使用PaddleLite提供的模型转换工具（model_optimize_tool）将量化模型转换成移动端预测的模型，然后加载转换后的模型进行预测部署。

###### 3.1 模型转换

参考[模型转换](../user_guides/model_optimize_tool)准备模型转换工具，建议从 Release 页面下载。

参考[模型转换](../user_guides/model_optimize_tool)使用模型转换工具，参数按照实际情况设置。比如在安卓手机 ARM 端进行预测，模型转换的命令为：
```bash
./OPT --model_dir=./mobilenet_v1_quant \
      --optimize_out_type=naive_buffer \
      --optimize_out=mobilenet_v1_quant_opt \
      --valid_targets=arm
```

###### 3.2 量化模型预测

和 FP32 模型一样，转换后的量化模型可以在 Android/IOS APP 中加载预测，建议参考[C++ Demo](./cpp_demo)、[Java Demo](./java_demo)、[Android/IOS Demo](../demo_guides/android_app_demo)。
