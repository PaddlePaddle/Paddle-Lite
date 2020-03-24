# PaddleLite使用NPU(华为)预测部署

Paddle Lite是首款支持华为自研达芬奇架构NPU（Kirin 810/990 SoC搭载的NPU）的预测框架。
原理是在线分析Paddle模型，将Paddle算子转成HiAI IR后，调用HiAI IR/Builder/Runtime APIs生成并执行HiAI模型。

## 已支持的设备

- 华为nova5、nova5i pro、mate30、mate30 pro、mate30 5G、荣耀v30，以及即将推出的mate40、p40。据华为透露，今后上市的大部分手机都会搭载其自研达芬奇架构NPU。

## 已支持的模型

- MobileNetV1
- MobileNetV2
- ResNet-18/50
- ShuffleNetV2
- CycleGAN (暂时需要华为内部rom的支持)
- 百度内部业务模型（由于涉密，不方便透露具体细节）

## 已支持（或部分支持）的Paddle算子

- sigmoid
- relu
- tanh
- relu_clipped
- leaky_relu
- softsign
- hard_sigmoid
- batch_norm
- concat
- conv2d
- depthwise_conv2d
- conv2d_transpose
- dropout
- elementwise_add
- elementwise_sub
- elementwise_mul
- elementwise_div
- fusion_elementwise_add_activation
- fusion_elementwise_sub_activation
- fusion_elementwise_mul_activation
- fusion_elementwise_div_activation
- fc
- bilinear_interp
- nearest_interp
- matmul
- mul
- pad2d
- pool2d
- reduce_mean
- reshape
- reshape2
- scale
- shuffle_channel
- softmax
- split
- sqrt
- square
- transpose
- transpose2
- unsqueeze
- unsqueeze2
- instance_norm (暂时需要华为内部rom的支持)
- layer_norm (暂时需要华为内部rom的支持)

## 编译支持NPU的Paddle Lite库

- 从https://developer.huawei.com/consumer/cn/hiai/下载华为HiAI DDK后解压到任意路径（注意：华为提供了多个版本的DDK，我们需要下载针对麒麟810/990芯片HiAI Foundation开发套件，例如最新的[DDK V310版本](https://obs.cn-north-2.myhwclouds.com/hms-ds-wf/sdk/hwhiai-ddk-100.310.011.010.zip)）。
- 将HiAI DDK中的ai_ddk_lib目录拷贝至Paddle Lite源码根目录后，使用[NPU编译脚本](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tools/build_npu.sh)编译full_publish和tiny_publish。

注意：以下是HiAI DDK V310版解压后的目录结构，需要将ai_ddk_lib目录拷贝至Paddle Lite源码根目录。
```shell
- app_sample
- ddk
  - ai_ddk_lib
    - include
    - lib # for armv7
    - lib64 # for armv8
- document
- tools
```

- full_publish and tiny_publish for armv8，由于HiAI DDK的armv7和armv8的so库均基于c++_shared构建，因此，建议使用c++_shared编译Paddle Lite。
```shell
$ ./lite/tools/build_npu.sh --arm_os=android --arm_abi=armv8 --arm_lang=gcc --android_stl=c++_shared full_publish
$ ./lite/tools/build_npu.sh --arm_os=android --arm_abi=armv8 --arm_lang=gcc --android_stl=c++_shared tiny_publish
```

- full_publish and tiny_publish for armv7
```shell
$ ./lite/tools/build_npu.sh --arm_os=android --arm_abi=armv7 --arm_lang=gcc --android_stl=c++_shared full_publish
$ ./lite/tools/build_npu.sh --arm_os=android --arm_abi=armv7 --arm_lang=gcc --android_stl=c++_shared tiny_publish
```

注意：为了保证编译环境一致，建议参考[源码编译](../user_guides/source_compile)中的Docker开发环境进行配置，然后再执行上述命令。

## 优化生成NPU模型

- model_optimize_tool工具已经支持生成NPU模型，仅需要将valid_targets设置为npu,arm即可，具体参考[模型转化方法](../user_guides/model_optimize_tool)。
```shell
./model_optimize_tool --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=(protobuf|naive_buffer) \
    --optimize_out=<output_optimize_model_dir> \
    --valid_targets=npu,arm \
    --record_tailoring_info =(true|false)
```
- model_optimize_tool生成的模型只是标记了NPU支持的Paddle算子，并没有真正生成NPU HiAI模型，只有在执行时才会将标记的Paddle算子转成HiAI IR，最终生成并执行HiAI模型，具体实现参考PR[2576](https://github.com/PaddlePaddle/Paddle-Lite/pull/2576)。
- 不同模型，不同型号（ROM版本）的华为手机，在执行阶段，由于某些Paddle算子无法完全转成HiAI IR，或目标手机的HiAI版本过低等原因，可能导致HiAI模型无法成功生成，在这种情况下，Paddle Lite会调用CPU版算子进行运算完成整个预测任务。

## 通过JAVA接口加载并执行NPU模型

- 使用方法和[Java实例](java_demo)一致，无需额外设置任何参数，只需将模型换成NPU模型即可。[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)中的Image Classification Demo for Android是同时支持CPU和NPU两种模型的图像分类Demo。

注意：在拷贝libpaddle_lite_jni.so的时候，由于依赖HiAI DDK so和libc++_shared.so库，需要将HiAI DDK中ai_ddk_lib/lib或ai_ddk_lib/lib64目录下的所有so和libc++_shared.so，拷到libpaddle_lite_jni.so同级目录下。

## 通过C++接口加载并执行NPU模型

- 使用方法和[C++实例](cpp_demo)一致，同样无需额外设置任何参数，只需将模型换成NPU模型即可。

注意：1）不能使用安卓模拟器，需要使用真实设备，且必须是支持NPU的华为手机。2）在使用adb push命令向手机推送目标程序时，需要将HiAI DDK中ai_ddk_lib/lib或ai_ddk_lib/lib64目录下的所有so和libc++_shared.so，推送到目标程序同级目录下。


## 其它说明

- 华为达芬奇架构的NPU内部大量采用float16进行运算，因此，预测结果会存在偏差，但大部分情况下精度不会有较大损失，可参考[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)中Image Classification Demo for Android对同一张图片CPU与NPU的预测结果。
- 华为Kirin 810/990 Soc搭载的自研达芬奇架构的NPU，与Kirin 970/980 Soc搭载的寒武纪NPU不一样，同样的，与Hi3559A、Hi3519A使用的NNIE也不一样，Paddle Lite只支持华为自研达芬奇架构NPU。
- 我们正在持续增加能够适配HiAI IR的Paddle算子bridge/converter，以便适配更多Paddle模型，同时华为研发同学也在持续对HiAI IR性能进行优化。
