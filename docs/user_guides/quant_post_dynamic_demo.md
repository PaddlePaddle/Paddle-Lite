# 动态离线量化完整示例

本章节介绍使用动态离线量化方法产出量化模型，使用 Paddle Lite 加载量化模型进行预测。

动态离线量化方法简单易用，不需要校准数据。主要用于减小模型体积，无法明显提升预测速度。
更多模型量化的介绍，请参考[量化训练文档](./quant_aware)、[动态离线量化文档](./quant_post_dynamic)和[静态离线量化文档](./quant_post_static)。

## 1 产出量化模型

### 1.1 准备 OPT 工具

参考[OPT 文档](./model_optimize_tool)，下载或者编译 OPT 工具，其中可执行文件 OPT 和 python 版本 OPT 都提供了动态图离线量化功能。

此处安装 2.9 版本 Paddle Lite:

```
pip install paddlelite==2.9 
```

### 1.2 产出优化后的量化模型

下载并解压[mobilenetv1 模型](https://paddle-inference-dist.cdn.bcebos.com/mobilenet_v1.tar.gz)。
```
wget https://paddle-inference-dist.cdn.bcebos.com/mobilenet_v1.tar.gz
tar zxf mobilenet_v1.tar.gz
```

使用 OPT 工具，产出优化后的非量化 mobilenetv1 模型。

```shell
paddle_lite_opt \
    --model_dir=mobilenet_v1 \
    --optimize_out_type=naive_buffer \
    --optimize_out=mobilenet_v1_opt \
    --valid_targets=arm
```

使用 OPT 工具，开启动态离线量化，设置量化为 16bit，产出优化后的 mobilenetv1 量化模型。

```shell
paddle_lite_opt \
    --model_dir=mobilenet_v1 \
    --optimize_out_type=naive_buffer \
    --optimize_out=mobilenet_v1_int16_opt \
    --valid_targets=arm  \
    --quant_model=true \
    --quant_type=QUANT_INT16
```

使用 OPT 工具，开启动态离线量化，设置量化为 8bit，产出优化后的 mobilenetv1 量化模型。

```shell
paddle_lite_opt \
    --model_dir=mobilenet_v1 \
    --optimize_out_type=naive_buffer \
    --optimize_out=mobilenet_v1_int8_opt \
    --valid_targets=arm  \
    --quant_model=true \
    --quant_type=QUANT_INT8
```

对比优化后的模型体积:
* mobilenet_v1_opt.nb 文件是 17M
* mobilenet_v1_int16_opt.nb 文件是 8.3M
* mobilenet_v1_int8_opt.nb 文件是 4.3M

## 2 部署量化模型

### 2.1 环境准备

因为需要执行示例，所以需要准备一台 armv7 或 armv8 架构的安卓手机。

### 2.2 编译 Android 预测库和示例

在 Paddle Lite 根目录，执行以下编译命令。

```
./lite/tools/build_android.sh --toolchain=gcc --with_extra=ON full_publish
```

在 Paddle Lite 根目录，进入示例文件。

```
cd build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/quant_post_dynamic
```

执行`prepare.sh`脚本，会编译可执行文件，同时将测试文件、预测库、测试脚本存放到`quant_post_dynamic_demo`文件夹。

```
sh prepare.sh
```

### 2.3 执行示例

(1) 设置手机

手机USB连接电脑，打开`设置 -> 开发者模式 -> USB调试 -> 允许（授权）当前电脑调试手机`。保证当前电脑已经安装[adb 工具](https://developer.android.com/studio/command-line/adb)，运行以下命令，确认当前手机设备已被识别：

``` shell
adb devices
# 如果手机设备已经被正确识别，将输出类似信息
List of devices attached
017QXM19C1000664	device
```

(2) 预测部署

将`quant_post_dynamic_demo`文件夹 push 到手机端。

```
adb push quant_post_dynamic_demo /data/local/tmp/quant_post_dynamic_demo
```

将优化好的模型 push 到手机端`quant_post_dynamic_demo`文件夹。

```
adb push mobilenet_v1_opt.nb /data/local/tmp/quant_post_dynamic_demo
adb push mobilenet_v1_int16_opt.nb /data/local/tmp/quant_post_dynamic_demo
adb push mobilenet_v1_int8_opt.nb /data/local/tmp/quant_post_dynamic_demo
```

基于相同的输入、预测库、可执行文件，加载量化前后模型进行预测，得到输出。

```
adb shell
cd /data/local/tmp/quant_post_dynamic_demo
sh run.sh
```

执行量化前的 mobilenetv1 模型，Log 信息如下：

```
max_value:0.936886
max_index:65
max_value_ground_truth:0.936887
max_index_ground_truth:65
----------Pass Test----------
```

执行量化后的 mobilenetv1_int16 模型，Log 信息如下：

```
max_value:0.936943
max_index:65
max_value_ground_truth:0.936887
max_index_ground_truth:65
----------Pass Test----------
```

执行量化后的 mobilenetv1_int8 模型，Log 信息如下：
```
max_value:0.937905
max_index:65
max_value_ground_truth:0.936887
max_index_ground_truth:65
----------Pass Test----------
```

从 Log 信息中可以发现，量化前后的模型分类结果相同，实际预测的类别概率和真实的类别概率，数值误差较小。
