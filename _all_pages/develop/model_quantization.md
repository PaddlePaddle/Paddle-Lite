---
layout: post
title: 模型量化
---
* TOC
{:toc}

Paddle-Lite支持加载运行[PaddlePaddle框架](https://github.com/PaddlePaddle/Paddle)量化训练产出的模型。本文主要介绍如何基于PaddlePaddle和Paddle-Lite对模型进行端到端的量化训练和推理执行。PaddlePaddle框架中所使用的量化训练原理请猛戳[此处](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/tutorial.md#1-quantization-aware-training%E9%87%8F%E5%8C%96%E4%BB%8B%E7%BB%8D)。如果您是初次接触PaddlePaddle框架，建议首先学习[新人入门](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/index_cn.html)和[使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/index_cn.html)。
> 备注：本文中所使用的模型量化示例均为MobileNetV1。

### 一、使用PaddleSlim模型压缩工具获取量化模型
**用户须知**: 现阶段的量化训练主要针对卷积层（包括二维卷积和Depthwise卷积）以及全连接层进行量化。卷积层和全连接层在PaddlePaddle框架中对应算子包括conv2d、depthwise_conv2d和mul等。量化训练会对conv2d、depthwise_conv2d和mul进行量化操作，且要求它们的输入中必须包括激活和参数两部分。

#### 1. 安装PaddlePaddle
根据操作系统、安装方式、Python版本和CUDA版本，按照[官方说明](https://paddlepaddle.org.cn/start)安装PaddlePaddle1.5.1版本。例如：

Ubuntu 16.04.4 LTS操作系统，CUDA9，cuDNN7，GPU版本安装:
```bash
pip install paddlepaddle-gpu==1.5.1.post97 -i https://mirrors.aliyun.com/pypi/simple/
```

Ubuntu 16.04.4 LTS操作系统，CPU版本安装:
```bash
pip install paddlepaddle==1.5.1 -i https://mirrors.aliyun.com/pypi/simple/
```

#### 2. 克隆量化训练所需的代码库
克隆[PaddlePaddle/models](https://github.com/PaddlePaddle/models)到本地，并进入models/PaddleSlim路径。执行如下命令：
```bash
git clone https://github.com/PaddlePaddle/models.git
cd models/PaddleSlim
```

#### 3. 数据准备
##### 3.1 训练数据准备
参考[models/PaddleCV/image_classification](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#data-preparation)下的数据准备教程准备训练数据，并放入PaddleSlim/data路径下。

##### 3.2 预训练模型准备

脚本run.sh会自动从[models/PaddleCV/image_classification](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#supported-models-and-performances)下载MobileNetV1的预训练模型，并放入PaddleSlim/pretrain路径下。

经过以上三步，PaddleSlim目录下的文件结构如下所示：

```bash
.
├── compress.py # 模型压缩任务主脚本，定义了压缩任务需要的模型相关信息
├── configs # 压缩任务的配置文件，包括:蒸馏、int8量化量化、filter剪切和组合策略的配置文件
├── data # 存放训练数据（需要用户自己创建）
│   └── ILSVRC2012
├── pretrain # 存放预训练模型参数，执行run.sh自动生成
│   ├── MobileNetV1_pretrained
│   ├── MobileNetV1_pretrained.tar
│   ├── ResNet50_pretrained
│   └── ResNet50_pretrained.tar
├── docs # 文档目录
├── light_nas
├── models # 模型网络结构的定义，如MobileNetV1
├── quant_low_level_api # 量化训练的底层API, 用于灵活定制量化训练的过程，适用于高阶用户
├── reader.py # 定义数据处理逻辑
├── README.md
├── run.sh # 模型压缩任务启动脚本
└── utility.py # 定义了常用的工具方法
```

#### 4. 压缩脚本介绍
在`compress.py`中定义了执行压缩任务需要的所有模型相关的信息，这里对几个关键的步骤进行简要介绍：

##### 4.1 目标网络的定义

compress.py的以下代码片段定义了train program, 这里train program只有前向计算操作。
```python
out = model.net(input=image, class_dim=args.class_dim)
cost = fluid.layers.cross_entropy(input=out, label=label)
avg_cost = fluid.layers.mean(x=cost)
acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
```

然后，通过clone方法得到eval_program, 用来在压缩过程中评估模型精度，如下：

```python
val_program = fluid.default_main_program().clone()
```

定义完目标网络结构，需要对其初始化，并根据需要加载预训练模型。

##### 4.2  定义feed_list和fetch_list
对于train program, 定义train_feed_list用于指定从train data reader中取的数据feed给哪些variable。定义train_fetch_list用于指定在训练时，需要在log中展示的结果。如果需要在训练过程中在log中打印accuracy信心，则将('acc_top1', acc_top1.name)添加到train_fetch_list中即可。
```python
train_feed_list = [('image', image.name), ('label', label.name)]
train_fetch_list = [('loss', avg_cost.name)]
```

> 注意： 在train_fetch_list里必须有loss这一项。

对于eval program. 同上定义eval_feed_list和train_fetch_list:

```python
val_feed_list = [('image', image.name), ('label', label.name)]
val_fetch_list = [('acc_top1', acc_top1.name), ('acc_top5', acc_top5.name)]
```

##### 4.3 Compressor和量化配置文件
I. `compress.py`主要使用Compressor和yaml文件完成对模型的量化训练工作。Compressor类的定义如下：
```python
class Compressor(object):
    def __init__(self,
                 place,
                 scope,
                 train_program,
                 train_reader=None,
                 train_feed_list=None,
                 train_fetch_list=None,
                 eval_program=None,
                 eval_reader=None,
                 eval_feed_list=None,
                 eval_fetch_list=None,
                 teacher_programs=[],
                 checkpoint_path='./checkpoints',
                 train_optimizer=None,
                 distiller_optimizer=None):
```

在定义Compressor对象时，需要注意以下问题：
* train program如果带反向operators和优化更新相关的operators, 参数train_optimizer需要设置为None.
* eval_program中parameter的名称需要与train_program中的parameter的名称完全一致。
* 最终保存的量化模型是在eval_program网络基础上进行剪枝保存的。所以，如果用户希望最终保存的模型可以用于inference, 则eval program需要包含推理阶段需要的各种operators.
* checkpoint保存的是float数据类型的模型。

II. `configs/quantization.yaml`量化配置文件示例如下：

```python
version: 1.0
strategies:
    quantization_strategy:
        class: 'QuantizationStrategy'
        start_epoch: 0
        end_epoch: 9
        float_model_save_path: './output/float'
        mobile_model_save_path: './output/mobile'
        int8_model_save_path: './output/int8'
        weight_bits: 8
        activation_bits: 8
        weight_quantize_type: 'abs_max'
        activation_quantize_type: 'moving_average_abs_max'
        save_in_nodes: ['image']
        save_out_nodes: ['fc_0.tmp_2']
compressor:
    epoch: 10
    checkpoint_path: './checkpoints_quan/'
    strategies:
        - quantization_strategy
```
其中，可配置参数包括：
- **class:** 量化策略的类名称，目前仅支持`QuantizationStrategy`。
- **start_epoch:** 在start_epoch开始之前，量化训练策略会往train_program和eval_program插入量化operators和反量化operators。 从start_epoch开始，进入量化训练阶段。
- **end_epoch:** 在end_epoch结束之后，会保存用户指定格式的模型。注意：end_epoch之后并不会停止量化训练，而是继续训练直到epoch数等于compressor.epoch值为止。举例来说，当start_epoch=0，end_epoch=0，compressor.epoch=2时，量化训练开始于epoch0，结束于epoch1，但保存的模型是epoch0结束时的参数状态。
- **float_model_save_path:**  保存float数据格式的模型路径，即该路径下的模型参数范围为int8范围但参数数据类型为float32。如果设置为None, 则不存储float格式的模型，默认为None。**注意：Paddle-Lite即使用该目录下的模型进行量化模型推理优化，详见本文[使用Paddle-Lite运行量化模型推理](#二使用Paddle-Lite运行量化模型推理)部分。**
- **int8_model_save_path:** 保存int8数据格式的模型路径，即该路径下的模型参数范围为int8范围且参数数据类型为int8。如果设置为None, 则不存储int8格式的模型，默认为None.
- **mobile_model_save_path:** 保存兼容paddle-mobile框架的模型路径。如果设置为None, 则不存储paddle-mobile格式的模型，默认为None。目前paddle-mobile已升级为Paddle-Lite。
- **weight_bits:** 量化weight的bit数，注意偏置(bias)参数不会被量化。
- **activation_bits:** 量化activation的bit数。
-  **weight_quantize_type:** weight量化方式，目前量化训练支持`abs_max`、 `channel_wise_abs_max`。
- **activation_quantize_type:** activation量化方式，目前量化训练支持`abs_max`、 `range_abs_max`和`moving_average_abs_max`。
- **save_in_nodes:** variable名称列表。在保存量化后模型的时候，需要根据save_in_nodes对eval programg 网络进行前向遍历剪枝。默认为eval_feed_list内指定的variable的名称列表。
- **save_out_nodes:** varibale名称列表。在保存量化后模型的时候，需要根据save_out_nodes对eval programg 网络进行回溯剪枝。默认为eval_fetch_list内指定的variable的名称列表。

> **备注：**
>
> 1）`abs_max`意为在训练的每个step及inference阶段均动态计算量化scale值。`channel_wise_abs_max`与`abs_max`类似，不同点在于它会对卷积权重进行分channel求取量化scale。换言之，`abs_max`属于tensor-wise量化，而`channel_wise_abs_max`属于channel-wise量化，详细说明请猛戳[此处](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/quantization/training_quantization_model_format.md)。
> 
> 2）`moving_average_abs_max`和`range_abs_max`意为在训练阶段计算出一个静态的量化scale值，并将其用于inference阶段。`moving_average_abs_max`使用窗口滑动平均的方法计算量化scale，而`range_abs_max`则使用窗口绝对值最大值的方式。
> 
> 3）**目前，Paddle-Lite仅支持运行weight量化方式使用`abs_max`且activation量化方式使用`moving_average_abs_max`或`range_abs_max`产出的量化模型**。

#### 5. 执行int8量化训练

修改run.sh，即注释掉`# enable GC strategy`与`# for sensitivity filter pruning`之间的内容并打开`#for quantization`相关的脚本命令（所需打开注释的命令如下所示）。

```bash
# for quantization
#---------------------------
export CUDA_VISIBLE_DEVICES=0
python compress.py \
--batch_size 64 \
--model "MobileNet" \
--pretrained_model ./pretrain/MobileNetV1_pretrained \
--compress_config ./configs/quantization.yaml \
--quant_only True
```
最后，运行`sh run.sh`命令开始int8量化训练。

### 二、使用Paddle-Lite运行量化模型推理
上述量化训练过程完成后，若用户按照本文中所述`configs/quantization.yaml`文件内容配置的模型输出路径，则可在models/PaddleSlim/output目录下看到`float`、`int8`和`mobile`三个目录，其中：
* float目录: 参数范围为int8范围但参数数据类型为float32的量化模型。Paddle-Lite即使用该目录下的模型文件及参数进行量化模型的部署。
* int8目录: 参数范围为int8范围且参数数据类型为int8的量化模型。
* mobile目录：参数特点与int8目录相同且兼容paddle-mobile的量化模型（目前paddle-mobile已升级为Paddle-Lite）。

#### 1. 在手机端准备量化模型文件
这里我们主要使用float目录下的模型文件（用户亦可以选择使用官方已经预训练好的MobileNetV1量化模型，[点击此处](https://paddle-inference-dist.bj.bcebos.com/int8%2Fpretrain%2Fmobilenet_v1_quant%2Ffloat.zip)进行下载）。
使用如下命令将float目录下的量化模型文件导入到手机端：

```bash
adb shell mkdir -p /data/local/tmp/mobilenet_v1_quant
adb push float/* /data/local/tmp/mobilenet_v1_quant
```

#### 2. 使用模型优化工具对量化模型进行优化
克隆[PaddlePaddle/Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)到本地（注意执行以下所有命令时均默认Paddle-Lite源码文件夹在当前目录下）。根据[Docker开发环境的配置说明文档](https://github.com/PaddlePaddle/Paddle-Lite/wiki/source_compile#1-docker%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83)准备Paddle-Lite编译环境。若用户按照文档配置docker编译环境，则进入docker容器可看到宿主机端的Paddle-Lite源码文件夹被映射挂载到容器的/Paddle-Lite目录下。在docker容器中执行以下编译命令：

```bash
cd /Paddle-Lite
./lite/tools/build.sh             \
   --arm_os=android               \
   --arm_abi=armv8                \
   --arm_lang=gcc                 \
   --android_stl=c++_static       \
   full_publish
```

* 编译完成后退出docker容器，模型优化工具model\_optimize\_tool在宿主机的存放位置为`Paddle-Lite/build.lite.android.armv8.gcc/lite/api/model_optimize_tool`。此时，目录结构如下所示：
```bash
Paddle-Lite/
|-- build.lite.android.armv8.gcc
|   |-- lite
|   |   |-- api
|   |   |   |-- model_optimize_tool
```
* 在宿主机执行如下命令将`model_optimize_tool`文件导入到手机端。

```bash
adb push Paddle-Lite/build.lite.android.armv8.gcc/lite/api/model_optimize_tool /data/local/tmp
```

* 在宿主机执行如下命令，完成对量化训练模型的优化，产生适合在移动端直接部署的量化模型。
```bash
adb shell rm -rf /data/local/tmp/mobilenet_v1_quant_opt
adb shell chmod +x /data/local/tmp/model_optimize_tool
adb shell /data/local/tmp/model_optimize_tool                 \
--model_file=/data/local/tmp/mobilenet_v1_quant/model         \
--param_file=/data/local/tmp/mobilenet_v1_quant/weights       \
--optimize_out_type=naive_buffer                              \
--optimize_out=/data/local/tmp/mobilenet_v1_quant_opt         \
--valid_targets=arm                                           \
--prefer_int8_kernel=true
```
model\_optimize\_tool的详细使用方法请猛戳[此处](https://github.com/PaddlePaddle/Paddle-Lite/wiki/model_optimize_tool#%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)。

> 备注：如前所述，量化训练后，float目录下的模型参数范围为int8，但参数数据类型仍为float32类型，仅这样确实没有起到模型参数压缩的效果。但是，经过model\_optimize\_tool工具优化后对应的量化参数均会以int8类型重新存储达到参数压缩的效果，且模型结构也被优化（如进行了各种operator fuse操作）。

#### 3. 使用mobilenetv1\_light\_api运行优化后的量化模型

在docker容器中执行如下命令获取Paddle-Lite轻量级API的demo：

```bash
cd /Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/mobile_light
make clean && make -j
```
执行完上述命令后退出docker容器，并可在宿主机`Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/mobile_light/`路径下看到`mobilenetv1_light_api`可执行文件。将`mobilenetv1_light_api`导入到手机端并运行量化模型推理。执行命令如下：

```bash
adb push Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/mobile_light/mobilenetv1_light_api /data/local/tmp
adb shell chmod +x /data/local/tmp/mobilenetv1_light_api
adb shell /data/local/tmp/mobilenetv1_light_api               \
    --model_dir=/data/local/tmp/mobilenet_v1_quant_opt
```
**程序运行结果如下：**
```bash
Output dim: 1000
Output[0]: 0.000228
Output[100]: 0.000260
Output[200]: 0.000250
Output[300]: 0.000560
Output[400]: 0.000950
Output[500]: 0.000275
Output[600]: 0.005143
Output[700]: 0.002509
Output[800]: 0.000538
Output[900]: 0.000969
```
在C++中使用Paddle-Lite API的方法请猛戳[此处](https://github.com/PaddlePaddle/Paddle-Lite/wiki/demos#如何在代码中使用-api)，用户也可参考[mobilenetv1_light_api.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/cxx/mobile_light/mobilenetv1_light_api.cc)的代码示例。

### 三、FAQ

**问题**：Compiled with WITH_GPU, but no GPU found in runtime

**解答**：检查本机是否支持GPU训练，如果不支持请使用CPU训练。如果在docker进行GPU训练，请使用nvidia_docker启动容器。

**问题**：Inufficient GPU memory to allocation. at [/paddle/paddle/fluid/platform/gpu_info.cc:262]
	
**解答**：正确设置run.sh脚本中`CUDA_VISIBLE_DEVICES`，确保显卡剩余内存大于需要内存。

