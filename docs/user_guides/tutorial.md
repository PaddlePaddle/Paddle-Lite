# 使用流程

Lite是一种轻量级、灵活性强、易于扩展的高性能的深度学习预测框架，它可以支持诸如ARM、OpenCL、NPU等等多种终端，同时拥有强大的图优化及预测加速能力。如果您希望将Lite框架集成到自己的项目中，那么只需要如下几步简单操作即可。

## 一. 准备模型

Lite框架目前支持的模型结构为[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)深度学习框架产出的模型格式。因此，在您开始使用 Lite 框架前您需要准备一个由PaddlePaddle框架保存的模型。
如果您手中的模型是由诸如Caffe2、Tensorflow等框架产出的，那么我们推荐您使用 [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) 工具进行模型格式转换。

## 二. 模型优化

Lite框架拥有强大的加速、优化策略及实现，其中包含诸如量化、子图融合、Kernel优选等等优化手段，为了方便您使用这些优化策略，我们提供了[opt](model_optimize_tool)帮助您轻松进行模型优化。优化后的模型更轻量级，耗费资源更少，并且执行速度也更快。

opt的详细介绍，请您参考 [模型优化方法](model_optimize_tool) 。

使用opt，您只需编译后在开发机上执行以下代码：

``` shell
$ cd <PaddleLite_base_path>
$ cd build.opt/lite/api/
$ ./opt \
    --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=(protobuf|naive_buffer) \
    --optimize_out=<output_optimize_model_dir> \
    --valid_targets=(arm|opencl|x86)
```

其中，optimize_out为您希望的优化模型的输出路径。optimize_out_type则可以指定输出模型的序列化方式，其目前支持Protobuf与Naive Buffer两种方式，其中Naive Buffer是一种更轻量级的序列化/反序列化实现。如果你需要使用Lite在mobile端进行预测，那么您需要设置optimize_out_type=naive_buffer。

## 三. 使用Lite框架执行预测

在上一节中，我们已经通过`opt`获取到了优化后的模型，使用优化模型进行预测也十分的简单。为了方便您的使用，Lite进行了良好的API设计，隐藏了大量您不需要投入时间研究的细节。您只需要简单的五步即可使用Lite在移动端完成预测（以C++ API进行说明）：


1. 声明MobileConfig。在config中可以设置**从文件加载模型**也可以设置**从memory加载模型**。从文件加载模型需要声明模型文件路径，如 `config.set_model_from_file(FLAGS_model_file)` ；从memory加载模型方法现只支持加载优化后模型的naive buffer，实现方法为：
`void set_model_from_buffer(model_buffer) `

2. 创建Predictor。Predictor即为Lite框架的预测引擎，为了方便您的使用我们提供了 `CreatePaddlePredictor` 接口，你只需要简单的执行一行代码即可完成预测引擎的初始化，`std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor(config)` 。
3. 准备输入。执行predictor->GetInput(0)您将会获得输入的第0个field，同样的，如果您的模型有多个输入，那您可以执行 `predictor->GetInput(i)` 来获取相应的输入变量。得到输入变量后您可以使用Resize方法指定其具体大小，并填入输入值。
4. 执行预测。您只需要执行 `predictor->Run()` 即可使用Lite框架完成预测。
5. 获取输出。与输入类似，您可以使用 `predictor->GetOutput(i)` 来获得输出的第i个变量。您可以通过其shape()方法获取输出变量的维度，通过 `data<T>()` 模板方法获取其输出值。




## 四. Lite API

为了方便您的使用，我们提供了C++、Java、Python三种API，并且提供了相应的api的完整使用示例:[C++完整示例](../demo_guides/cpp_demo)、[Java完整示例](../demo_guides/java_demo)、[Python完整示例](../demo_guides/cuda)，您可以参考示例中的说明快速了解C++/Java/Python的API使用方法，并集成到您自己的项目中去。需要说明的是，为了减少第三方库的依赖、提高Lite预测框架的通用性，在移动端使用Lite API您需要准备Naive Buffer存储格式的模型，具体方法可参考第2节`模型优化`。

## 五. 测试工具

为了使您更好的了解并使用Lite框架，我们向有进一步使用需求的用户开放了 [Debug工具](debug#debug) 和 [Profile工具](debug#profiler)。Lite Model Debug Tool可以用来查找Lite框架与PaddlePaddle框架在执行预测时模型中的对应变量值是否有差异，进一步快速定位问题Op，方便复现与排查问题。Profile Monitor Tool可以帮助您了解每个Op的执行时间消耗，其会自动统计Op执行的次数，最长、最短、平均执行时间等等信息，为性能调优做一个基础参考。您可以通过 [相关专题](debug) 了解更多内容。
