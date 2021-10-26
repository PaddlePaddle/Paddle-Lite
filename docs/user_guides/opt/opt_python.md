
## python 调用 opt 转化模型

安装了 Paddle Lite 的 python 库后，可以通过 python 调用 opt 工具转化模型。（支持 MAC & Ubuntu 系统）

### 安装 Paddle Lite

```
# 当前最新版本是 2.9
pip install paddlelite==2.9
```

![install](https://paddlelite-data.bj.bcebos.com/doc_images/opt/install.gif)

### 帮助信息

安装成功后可以查看帮助信息
```bash
 paddle_lite_opt
```
![paddle_lite_opt](https://paddlelite-data.bj.bcebos.com/doc_images/opt/paddle_lite_opt.gif)

### 功能一：转化模型为 Paddle Lite 格式
opt 可以将 Paddle 原生模型转化为 Paddle Lite 支持的移动端模型：

- 存储格式转换，有效降低模型体积
- 执行“量化、子图融合、混合调度、 Kernel 优选”等优化操作，降低运行耗时与内存消耗

(1) 准备待优化的 PaddlePaddle 模型

- opt 支持下列5种模型格式
  - 用 `--model_dir=` 指定模型文件夹位置

```
# contents in model directory should be in one of these formats:
(1) __model__ + var1 + var2 + etc.
(2) model + var1 + var2 + etc.
(3) model.pdmodel + model.pdiparams
(4) model + params
(5) model + weights
```

- 其他格式： 
  - 用`--model_file=` 指定模型文件位置
  - 用`--param_file=` 指定参数文件位置
  
```
eg. model + param
# 加载这种非标准格式时： 需要指定 模型和参数文件 位置
paddle_lite_opt --model_file=./model --param_file=./param
```

(2) 终端中执行`opt`命令转化模型
**使用示例**：转化`mobilenet_v1`模型

```
paddle_lite_opt --model_dir=./mobilenet_v1 \
      --valid_targets=arm \
      --optimize_out=mobilenet_v1_opt
```
以上命令可将`mobilenet_v1`转化为arm平台模型，优化后的模型文件是`mobilenet_v1_opt.nb`：

![trans](https://paddlelite-data.bj.bcebos.com/doc_images/opt/trans.gif)



**注意**：若转化失败，提示模型格式不正确时

- 用`--model_file=` 指定模型文件位置
- 用`--param_file=` 指定参数文件位置

![other_type](https://paddlelite-data.bj.bcebos.com/doc_images/opt/other_type_trans.gif)


(3) **更详尽的转化命令**总结：

```shell
paddle_lite_opt \
    --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=(protobuf|naive_buffer) \
    --optimize_out=<output_optimize_model_dir> \
    --valid_targets=(arm|opencl|x86|npu|xpu|huawei_ascend_npu|imagination_nna|intel_fpga)\
    --enable_fp16=(true|false) \
    --quant_model=(true|false) \
    --quant_type=(QUANT_INT16|QUANT_INT8) 
```

| 选项         | 说明 |
| ------------------- | ------------------------------------------------------------ |
| --model_dir         | 待优化的 PaddlePaddle 模型（非 combined 形式）的路径 |
| --model_file        | 待优化的 PaddlePaddle 模型（ combined 形式）的网络结构文件路径。 |
| --param_file        | 待优化的 PaddlePaddle 模型（ combined 形式）的权重文件路径。 |
| --optimize_out_type | 输出模型类型，目前支持两种类型：protobuf 和 naive_buffer，其中 naive_buffer 是一种更轻量级的序列化/反序列化实现。若您需要在 mobile 端执行模型预测，请将此选项设置为 naive_buffer。默认为 protobuf。 |
| --optimize_out      | 优化模型的输出路径。                                         |
| --valid_targets     | 指定模型可执行的 backend，默认为 arm。可以同时指定多个 backend (以逗号分隔)，opt 将会自动选择最佳方式。如果需要支持华为 NPU（Kirin 810/990 Soc 搭载的达芬奇架构 NPU），应当设置为 "npu,arm"。 |
| --enable_fp16       | 设置是否使用 opt 中的 Float16 低精度量化功能，Float16 量化会提高速度提高、降低内存占用，但预测精度会有降低 |
| --quant_model       | 设置是否使用 opt 中的动态离线量化功能。 |
| --quant_type        | 指定 opt 中动态离线量化功能的量化类型，可以设置为 QUANT_INT8 和 QUANT_INT16，即分别量化为8比特和16比特。 量化为 int8 对模型精度有一点影响，模型体积大概减小4倍。量化为 int16 对模型精度基本没有影，模型体积大概减小2倍。|

* 如果待优化的 fluid 模型是非 combined 形式，请设置`--model_dir`，忽略`--model_file`和`--param_file`。
* 如果待优化的 fluid 模型是 combined 形式，请设置`--model_file`和`--param_file`，忽略`--model_dir`。
* `naive_buffer`的优化后模型为以`.nb`名称结尾的单个文件。
* `protobuf`的优化后模型为文件夹下的`model`和`params`两个文件。将`model`重命名为`__model__`用[Netron](https://lutzroeder.github.io/netron/)打开，即可查看优化后的模型结构。
* 删除`prefer_int8_kernel`的输入参数，`opt`自动判别是否是量化模型，进行相应的优化操作。
* `opt`中的动态离线量化功能和`PaddleSlim`中动态离线量化功能相同，`opt`提供该功能是为了用户方便使用。

### 功能二：统计模型算子信息、判断是否支持

opt 可以统计并打印出 model 中的算子信息、判断 Paddle Lite 是否支持该模型。并可以打印出当前 Paddle Lite 的算子支持情况。

（1）使用 opt 统计模型中算子信息

下面命令可以打印出 mobilenet_v1 模型中包含的所有算子，并判断在硬件平台`valid_targets`下 Paddle Lite 是否支持该模型

`paddle_lite_opt --print_model_ops=true  --model_dir=mobilenet_v1 --valid_targets=arm`

![opt_print_modelops](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/python_opt/check_model.png)

（2）使用 opt 打印当前 Paddle Lite 支持的算子信息

`paddle_lite_opt --print_all_ops=true`

以上命令可以打印出当前 Paddle Lite 支持的所有算子信息，包括 OP 的数量和每个 OP 支持哪些硬件平台：

![opt_print_allops](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/python_opt/print_op.png)

`paddle_lite_opt --print_supported_ops=true  --valid_targets=x86`

以上命令可以打印出当`valid_targets=x86`时 Paddle Lite 支持的所有 OP ：

![opt_print_supportedops](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/python_opt/print_x86op.png)
