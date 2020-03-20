
# 模型优化工具 opt

Paddle-Lite 提供了多种策略来自动优化原始的训练模型，其中包括量化、子图融合、混合调度、Kernel优选等等方法。为了使优化过程更加方便易用，我们提供了**opt** 工具来自动完成优化步骤，输出一个轻量的、最优的可执行模型。

具体使用方法介绍如下：

**注意**：`v2.2.0` 之前的模型转化工具名称为`model_optimize_tool`，从 `v2.3` 开始模型转化工具名称修改为 `opt`

## 准备opt
当前获得opt方法有三种：

1. **推荐！** 可以进入Paddle-Lite Github仓库的[release界面](https://github.com/PaddlePaddle/Paddle-Lite/releases)，选择release版本下载对应的转化工具`opt`    
   (release/v2.2.0之前的转化工具为model_optimize_tool、release/v2.3.0之后为opt)
2. 本文提供`release/v2.3`和`release/v2.2.0`版本的优化工具下载

|版本 | Linux | MacOS|
|---|---|---|
| `release/v2.3`| [opt](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/opt) | [opt_mac](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/opt_mac) |
|`release/v2.2.0`  | [model_optimize_tool](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/model_optimize_tool) | [model_optimize_tool_mac](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/model_optimize_tool_mac) |


3. 如果 release 列表里的工具不符合您的环境，可以下载Paddle-Lite 源码，源码编译出opt工具
```bash
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git checkout <release-version-tag>
./lite/tools/build.sh build_optimize_tool
```
编译结果位于`Paddle-Lite/build.opt/lite/api/opt`
**注意**：从源码编译opt前需要先[安装Paddle-Lite的开发环境](source_compile)。

## 使用opt

opt是 x86 平台上的可执行文件，需要在PC端运行：支持Linux终端和Mac终端。

### 帮助信息
 执行opt时不加入任何输入选项，会输出帮助信息，提示当前支持的选项：
```bash
 ./opt
```
![](https://paddlelite-data.bj.bcebos.com/doc_images/1.png)

### 功能一：转化模型为Paddle-Lite格式
opt可以将PaddlePaddle的部署模型格式转化为Paddle-Lite 支持的模型格式，期间执行的操作包括：

- 将protobuf格式的模型文件转化为naive_buffer格式的模型文件，有效降低模型体积
- 执行“量化、子图融合、混合调度、Kernel优选”等图优化操作，提升其在Paddle-Lite上的运行速度、内存占用等效果

模型优化过程：

（1）准备待优化的PaddlePaddle模型

PaddlePaddle模型有两种保存格式：
   Combined Param：所有参数信息保存在单个文件`params`中，模型的拓扑信息保存在`__model__`文件中。

![opt_combined_model](https://paddlelite-data.bj.bcebos.com/doc_images%2Fcombined_model.png)

   Seperated Param：参数信息分开保存在多个参数文件中，模型的拓扑信息保存在`__model__`文件中。
![opt_seperated_model](https://paddlelite-data.bj.bcebos.com/doc_images%2Fseperated_model.png)

(2) 终端中执行`opt`优化模型
**使用示例**：转化`mobilenet_v1`模型

```
./opt --model_dir=./mobilenet_v1 \
      --valid_targets=arm \
      --optimize_out_type=naive_buffer \
      --optimize_out=mobilenet_v1_opt
```
以上命令可以将`mobilenet_v1`模型转化为arm硬件平台、naive_buffer格式的Paddle_Lite支持模型，优化后的模型文件为`mobilenet_v1_opt.nb`，转化结果如下图所示：

![opt_resulted_model](https://paddlelite-data.bj.bcebos.com/doc_images/2.png)


(3) **更详尽的转化命令**总结：

```shell
./opt \
    --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=(protobuf|naive_buffer) \
    --optimize_out=<output_optimize_model_dir> \
    --valid_targets=(arm|opencl|x86|npu|xpu) \
    --record_tailoring_info =(true|false)
```

| 选项         | 说明 |
| ------------------- | ------------------------------------------------------------ |
| --model_dir         | 待优化的PaddlePaddle模型（非combined形式）的路径 |
| --model_file        | 待优化的PaddlePaddle模型（combined形式）的网络结构文件路径。 |
| --param_file        | 待优化的PaddlePaddle模型（combined形式）的权重文件路径。 |
| --optimize_out_type | 输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf。 |
| --optimize_out      | 优化模型的输出路径。                                         |
| --valid_targets     | 指定模型可执行的backend，默认为arm。目前可支持x86、arm、opencl、npu、xpu，可以同时指定多个backend(以空格分隔)，Model Optimize Tool将会自动选择最佳方式。如果需要支持华为NPU（Kirin 810/990 Soc搭载的达芬奇架构NPU），应当设置为npu, arm。 |
| --record_tailoring_info | 当使用 [根据模型裁剪库文件](./library_tailoring.html) 功能时，则设置该选项为true，以记录优化后模型含有的kernel和OP信息，默认为false。 |

* 如果待优化的fluid模型是非combined形式，请设置`--model_dir`，忽略`--model_file`和`--param_file`。
* 如果待优化的fluid模型是combined形式，请设置`--model_file`和`--param_file`，忽略`--model_dir`。
* 优化后的模型为以`.nb`名称结尾的单个文件。
* 删除`prefer_int8_kernel`的输入参数，`opt`自动判别是否是量化模型，进行相应的优化操作。

### 功能二：统计模型算子信息、判断是否支持

opt可以统计并打印出model中的算子信息、判断Paddle-Lite是否支持该模型。并可以打印出当前Paddle-Lite的算子支持情况。

（1）使用opt统计模型中算子信息

下面命令可以打印出mobilenet_v1模型中包含的所有算子，并判断在硬件平台`valid_targets`下Paddle-Lite是否支持该模型

`./opt --print_model_ops=true  --model_dir=mobilenet_v1 --valid_targets=arm`

![opt_print_modelops](https://paddlelite-data.bj.bcebos.com/doc_images/3.png)

（2）使用opt打印当前Paddle-Lite支持的算子信息

`./opt --print_all_ops=true`

以上命令可以打印出当前Paddle-Lite支持的所有算子信息，包括OP的数量和每个OP支持哪些硬件平台：

![opt_print_allops](https://paddlelite-data.bj.bcebos.com/doc_images/4.png)

`./opt ----print_supported_ops=true  --valid_targets=x86`

以上命令可以打印出当`valid_targets=x86`时Paddle-Lite支持的所有OP：

![opt_print_supportedops](https://paddlelite-data.bj.bcebos.com/doc_images/5.png)

## 其他功能：合并x2paddle和opt的一键脚本

**背景**：如果想用Paddle-Lite运行第三方来源（tensorflow、caffe、onnx）模型，一般需要经过两次转化。即使用x2paddle工具将第三方模型转化为PaddlePaddle格式，再使用opt将PaddlePaddle模型转化为Padde-Lite可支持格式。
为了简化这一过程，我们提供一键脚本，将x2paddle转化和opt转化合并：

**一键转化脚本**：[auto_transform.sh](https://github.com/PaddlePaddle/Paddle-Lite/blob/release/v2.3/lite/tools/auto_transform.sh)


**环境要求**：使用`auto_transform.sh`脚本转化第三方模型时，需要先安装x2paddle环境，请参考[x2paddle环境安装方法](https://github.com/PaddlePaddle/X2Paddle#环境依赖) 安装x2paddle和x2paddle依赖项(tensorflow、caffe等)。

**使用方法**：

（1）打印帮助帮助信息：` sh ./auto_transform.sh`

（2）转化模型方法

```bash
USAGE:
    auto_transform.sh combines the function of x2paddle and opt, it can 
    tranform model from tensorflow/caffe/onnx form into paddle-lite naive-buffer form.
----------------------------------------
example:
    sh ./auto_transform.sh --framework=tensorflow --model=tf_model.pb --optimize_out=opt_model_result
----------------------------------------
Arguments about x2paddle:
    --framework=(tensorflow|caffe|onnx);
    --model='model file for tensorflow or onnx';
    --prototxt='proto file for caffe' --weight='weight file for caffe'
 For TensorFlow:
   --framework=tensorflow --model=tf_model.pb

 For Caffe:
   --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel

 For ONNX
   --framework=onnx --model=onnx_model.onnx

Arguments about opt:
    --valid_targets=(arm|opencl|x86|npu|xpu); valid targets on Paddle-Lite.
    --fluid_save_dir='path to outputed model after x2paddle'
    --optimize_out='path to outputed Paddle-Lite model'
----------------------------------------
```
