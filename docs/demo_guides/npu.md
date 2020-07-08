# PaddleLite使用NPU(华为)预测部署

Paddle Lite是首款支持华为自研达芬奇架构NPU（Kirin 810/990 SoC搭载的NPU）的预测框架。
原理是在线分析Paddle模型，将Paddle算子转成HiAI IR后，调用HiAI IR/Builder/Runtime APIs生成并执行HiAI模型。

## 已支持的设备

- 华为nova5、nova5i pro、mate30、mate30 pro、mate30 5G、荣耀v30、p40、p40 pro，以及即将推出的mate40、。据华为透露，今后上市的大部分手机都会搭载其自研达芬奇架构NPU。

## 已支持的模型

- MobileNetV1
- MobileNetV2
- ResNet-18/50
- ShuffleNetV2
- squeezenet
- mnasnet
- yolov3
- CycleGAN (暂时需要华为内部rom的支持)
- 百度内部业务模型（由于涉密，不方便透露具体细节）

*CPU/NPU混合调度在部分模型可以获得更佳的性能*

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

- 从[华为HiAI平台](https://developer.huawei.com/consumer/cn/hiai)下载华为HiAI DDK后解压到任意路径（注意：华为提供了多个版本的DDK，我们需要下载针对麒麟810/990芯片HiAI Foundation开发套件，例如[DDK V310版本](https://obs.cn-north-2.myhwclouds.com/hms-ds-wf/sdk/hwhiai-ddk-100.310.011.010.zip)）。
- 将HiAI DDK中的ai_ddk_lib目录拷贝至Paddle Lite源码根目录后，使用[编译脚本](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tools/build_android.sh)编译 (需要指定NPU相关选项)。

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

- 推荐编译命令。由于HiAI DDK的so库均基于c++_shared构建，因此，建议使用c++_shared编译Paddle Lite。
```shell
# huawei_kirin_npu_sdk_root 需要指向 ai_ddk_lib 的路径
$ ./lite/tools/build_android.sh --android_stl=c++_shared --with_huawei_kirin_npu=ON --huawei_kirin_npu_sdk_root=<path-to-ai_ddk_lib>
# 其它选项可以通过 "./lite/tools/build_android.sh help" 查看，例如arm版本等 
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

**注意：由于华为手机root权限限制，现在仅支持JAVA接口加载和执行NPU模型**

- 使用方法和[Java实例](java_demo)一致，无需额外设置任何参数，只需将模型换成NPU模型即可。[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)中的Image Classification Demo for Android是同时支持CPU和NPU两种模型的图像分类Demo。

注意：在拷贝libpaddle_lite_jni.so的时候，由于依赖HiAI DDK so和libc++_shared.so库，需要将HiAI DDK中ai_ddk_lib/lib或ai_ddk_lib/lib64目录下的所有so和libc++_shared.so，拷到libpaddle_lite_jni.so同级目录下。

## 其它说明

- 华为达芬奇架构的NPU内部大量采用float16进行运算，因此，预测结果会存在偏差，但大部分情况下精度不会有较大损失，可参考[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)中Image Classification Demo for Android对同一张图片CPU与NPU的预测结果。
- 华为Kirin 810/990 Soc搭载的自研达芬奇架构的NPU，与Kirin 970/980 Soc搭载的寒武纪NPU不一样，同样的，与Hi3559A、Hi3519A使用的NNIE也不一样，Paddle Lite只支持华为自研达芬奇架构NPU。
- 我们正在持续增加能够适配HiAI IR的Paddle算子bridge/converter，以便适配更多Paddle模型，同时华为研发同学也在持续对HiAI IR性能进行优化。


## 手动分割子图

### 背景
- Paddle-Lite已经支持了大量的华为NPU的算子，但是仍然不能满足所有模型的需求。对于一个有部分算子不支持的模型，Paddle-Lite会将模型划分为可以跑在NPU上的子图和跑在CPU上的子图，实现NPU和CPU自动调度的功能，通常情况下可以获得比较好的性能。在一些特殊情况下，模型会被自动划分为比较多的子图，导致CPU和NPU的切换开销很大，从而导致整体性能变差。因此，需要手动分割子图的功能来指定一些算子跑在CPU上，避免子图过多。

### 功能
- 通过配置文件来指定需要强制跑在CPU上的算子

### 使用方法
- 1、通过netron打开paddle模型文件，可以查看模型结构，获得算子的类型、输入名称。输出名称。
    - 注意：Paddle-Lite会对模型进行优化，模型算子可以改变，需要以优化后的模型算子为准。后面会举例说明。
- 2、生成配置文件 ```split_cfg.txt```，记录需要跑在CPU上的算子信息。
    - 每行一条OP记录信息，以冒号":"分隔"op名称"，"op输入名"，"op输出名"，以逗号","分隔"op输入名"和"op输出名"中的不同var名。
    - 可以部分省略输入或者输出名。比如：```op3:in3_var0```表示，指定类型为"op3"，输入为"in3_var0"的算子；```op4```表示所有类型为"op4"的算子
    - 例子1：
    ```
    op0:in0_var0,in0_var1:out0_var0,out0_var1
    op1:in1_var0,in1_var1:out1_var0
    op2::out2_var0
    op3:in3_var0
    op4
    ```
    - 例子2：
    ```
    transpose:conv2d_22.tmp_1:transpose_0.tmp_0
    ```
    ![image](https://user-images.githubusercontent.com/50474132/80475316-4a5fda80-897b-11ea-910a-6aee13243387.png)

- 3、使用环境变量```SUBGRAPH_CUSTOM_PARTITION_CONFIG_FILE```指定配置文件的位置。
    - 例如：
    ```
    export SUBGRAPH_CUSTOM_PARTITION_CONFIG_FILE=/data/local/tmp/split_sfg.txt
    ```
- 4、以上步骤完成后，运行的模型中符合条件的算子将被强制跑在CPU上。

### 举例
- 以模型[image](https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_fluid.tar.gz)为例

- 1、可以使用netron查看模型

- 2、初步分析

    - 下图是ssd_mobilenet_v1中的部分结构。其中红色部分暂时不支持在NPU上运行，蓝色部分可能NPU上的性能不理想。此时，如果直接让预测库自动调度的话，可能会分成多个子图，而且整体性能不佳。因此，可以将蓝色部分和绿色部分整体指定在CPU上运行，让其他部分自动运行在NPU上(红色部分会自动在CPU上运行)。
    ![](https://user-images.githubusercontent.com/50474132/80453173-525b5280-895a-11ea-847f-c7dd5b5799de.png)

- 3、使用opt转换模型

    - opt转换过程中会打印log信息。在log中搜索```digraph G```和```// end G```可以找到优化后的模型图。
    ![](https://user-images.githubusercontent.com/50474132/80454098-145f2e00-895c-11ea-9f16-dde1483a9beb.png)
    ![](https://user-images.githubusercontent.com/50474132/80454123-1de89600-895c-11ea-86b9-a62d78a6616d.png)
    - 将从```digraph G```开始的，到```// end G```结束的整段模型图信息，保存到```.dot```格式的文件中。可以用```graphviz```打开查看，或者在[网页版](http://dreampuf.github.io/GraphvizOnline/)查看。
    ![](https://user-images.githubusercontent.com/50474132/80454841-47ee8800-895d-11ea-9531-5689c5560fcb.png)
    - 在此处确认需要被指定的算子是否被优化了。(期望是被指定的算子都还独立存在，如果被融合为了一个算子，需要指定此时融合后的算子)。

- 4、写配置文件

    - 在配置文件中指定可以支持NPU但是需要指定在CPU上运行的算子。
    ```
    reshape
    transpose
    concat
    softmax
    ```
    - 由于这些算子都指定在CPU上运行，因此不需要特意配置算子的输入输出名称。

- 5、指定配置文件路径

    - 通过```export SUBGRAPH_CUSTOM_PARTITION_CONFIG_FILE=your_split_config_file```的方式实现。

- 6、性能测试

    - 设备：华为mate30 5G
    - HIAI ddk版本：320
    - 性能：CPU约71.8ms，NPU约16.6ms。
    
