# 动态离线量化完整示例

本章节介绍使用动态离线量化方法产出量化模型，使用Paddle-Lite加载量化模型进行预测。

动态离线量化方法简单易用，不需要校准数据，主要用于减小模型体积，无法明显提升预测速度。更多模型量化的介绍，请参考[量化训练文档](../user_guides/quant_aware)，[静态离线量化文档](../user_guides/quant_post_static)，[动态离线量化文档](../user_guides/quant_post_dynamic)。

## 1 Paddle 产出量化模型

如果希望快速复现示例，可以暂时跳过产出模型的步骤，直接下载并解压[quant_post_dynamic.tar.gz](https://paddle-inference-dist.cdn.bcebos.com/PaddleLiteDemo/quant_post_dynamic.tar.gz)文件，其中包括原始模型、量化模型等。

### 1.1 环境准备

需要准备一台可以成功安装并执行PaddlePaddle的电脑。

### 1.2 产出量化模型

目前，我们需要使用PaddlePaddle和PaddleSlim提供的动态离线量化方法，产出量化模型。
后续会计划将该方法集成到Paddle-Lite 模型优化工具中，让大家更加方便使用。

参考[链接](https://www.paddlepaddle.org.cn/install/quick)安装最新版PaddlePaddle，可以是CPU或者GPU版本。

参考[链接](https://paddleslim.readthedocs.io/zh_CN/latest/install.html)安装最新版PaddleSlim。

下载[quant_post_dynamic.tar.gz](https://paddle-inference-dist.cdn.bcebos.com/PaddleLiteDemo/quant_post_dynamic.tar.gz)文件，其中包括待量化的模型、量化代码等。

如果安装的PaddlePaddle版本不低于2.0，修改下载文件夹中的`quant_post_dynamic.py`，即删除`paddle.enable_static()`的注释.

执行 `python quant_post_dynamic.py`，动态离线量化的`mobilenetv1`模型保存在`mobilenet_v1_int16`，其中权重是量化为16比特，量化的模型可以用于PaddleLite预测部署。

对比量化前后的模型体积，量化前模型是18M，量化后是8.9M。

## 2 Paddle-Lite 部署量化模型

### 2.1 环境准备

因为需要执行示例，所以需要准备一台armv7或armv8架构的安卓手机。

下述步骤中，如果要编译模型优化工具opt、预测库和示例，需要参考[源码编译环境准备](../source_compile/compile_env)，准备一台可以编译Paddle-Lite的电脑。
如果直接下载模型优化工具、预测库和示例，则不需要准备编译环境。

### 2.2 模型优化

如果希望快速复现示例，可以跳过模型优化步骤，上述下载的[quant_post_dynamic.tar.gz](https://paddle-inference-dist.cdn.bcebos.com/PaddleLiteDemo/quant_post_dynamic.tar.gz)文件中有优化好的模型文件，即是`mobilenet_v1_opt.nb`和`mobilenet_v1_int16_opt.nb`。

Paddle-Lite部署模型前，需要将原始模型进行优化，更多介绍请参考[模型优化工具 opt](../user_guides/model_optimize_tool)。

下面以Paddle-Lite release/v2.7分支为例，介绍编译opt工具，并且使用opt工具对原始模型进行转换。

在可以编译Paddle-Lite的电脑上，拉取Paddle-Lite的代码，切换到release/v2.7分支。
```
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git fetch origin release/v2.7:release/v2.7
git checkout release/v2.7
```

在Paddle-Lite根目录，执行编译命令，编译opt工具。编译完成后，opt工具在 `build.opt/lite/api/opt`。
```
./lite/tools/build.sh build_optimize_tool
```

在`quant_post_dynamic`文件夹下，使用opt工具执行模型转换，得到`mobilenet_v1_opt.nb`和`mobilenet_v1_int16_opt.nb`模型文件，两个模型的大小分别是17M和8.3M。
```
./opt --model_dir mobilenet_v1 --optimize_out_type naive_buffer --optimize_out mobilenet_v1_opt --valid_targets arm
./opt --model_dir mobilenet_v1_int16/quantized_model --optimize_out_type naive_buffer --optimize_out mobilenet_v1_int16_opt --valid_targets arm
```

### 2.3 编译Android预测库和示例

如果希望快速复现示例，可以跳过编译步骤，直接下载[quant_post_dynamic_demo.tar.gz](https://paddle-inference-dist.cdn.bcebos.com/PaddleLiteDemo/quant_post_dynamic_demo.tar.gz)文件，其中包括测试文件、预测库、测试脚本等。

在Paddle-Lite根目录，执行编译命令。
```
./lite/tools/build_android.sh --toolchain=gcc --with_extra=ON full_publish
```

在Paddle-Lite根目录，进入示例文件。
```
cd build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/quant_post_dynamic
```

执行`prepare.sh`脚本，会编译可执行文件，同时将测试文件、预测库、测试脚本存放到`quant_post_dynamic_demo`文件夹。
```
sh prepare.sh
```

### 2.4 执行示例

(1) 设置手机

手机USB连接电脑，打开`设置 -> 开发者模式 -> USB调试 -> 允许（授权）当前电脑调试手机`。保证当前电脑已经安装[adb工具](https://developer.android.com/studio/command-line/adb)，运行以下命令，确认当前手机设备已被识别：

``` shell
adb devices
# 如果手机设备已经被正确识别，将输出类似信息
List of devices attached
017QXM19C1000664	device
```

(2) 预测部署

`quant_post_dynamic`文件夹保存了上述转换好的模型、输入数据，`quant_post_dynamic_demo`文件夹保存了测试文件、预测库、测试脚本。

将`quant_post_dynamic_demo`和`quant_post_dynamic`文件夹push到手机端。
```
adb push quant_post_dynamic_demo /data/local/tmp 
adb push quant_post_dynamic /data/local/tmp/quant_post_dynamic_demo
```

基于相同的输入、预测库、可执行文件，加载量化前后模型进行预测，得到输出。
```
adb shell
cd /data/local/tmp/quant_post_dynamic_demo
sh run.sh
```

执行量化前的mobilenetv1模型，log信息如下。
```
max_value:0.936887
max_index:65
max_value_ground_truth:0.936887
max_index_ground_truth:65
----------Pass Test----------
```

执行量化后的mobilenetv1_int16模型，log信息如下。
```
max_value:0.936941
max_index:65
max_value_ground_truth:0.936887
max_index_ground_truth:65
----------Pass Test----------
```

从log信息中可以发现，量化前后的模型输出基本一致，数值误差极小。
