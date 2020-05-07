# 模型量化-量化训练

本文主要介绍使用Paddle-Lite加载PaddlePaddle产出的量化模型，并进行推理执行。

## 1 简介

量化训练是使用较多练数据，对训练好的预测模型进行量化。该方法使用模拟量化的思想，在训练阶段更新权重，实现减小量化误差。

使用条件：
* 有预训练模型
* 有较多训练数据（大于5000）

使用步骤：
* 产出量化模型：使用PaddlePaddle调用量化训练接口，产出量化模型
* 量化模型预测：使用PaddleLite加载量化模型进行预测推理

优点：
* 减小计算量、降低计算内存、减小模型大小
* 模型精度受量化影响小

缺点：
* 使用条件较苛刻，使用门槛稍高

建议首先使用“有校准数据训练后量化”对模型进行量化，然后使用使用量化模型进行预测。如果该量化模型的精度达不到要求，再使用“量化训练”。

## 2 产出量化模型

目前，PaddleSlim 框架的量化训练主要针对卷积层（包括二维卷积和Depthwise卷积）、和全连接层，对应算子是conv2d、depthwise_conv2d和mul。Paddle-Lite支持运行PaddlePaddle框架量化训练产出的模型，可以进一步加快模型在移动端的执行速度。

温馨提示：如果您是初次接触PaddlePaddle框架，建议首先学习[新人入门](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/index_cn.html)和[使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/index_cn.html)。

使用PaddleSlim模型压缩工具训练量化模型，请参考文档：
* 量化训练[快速开始教程](https://paddlepaddle.github.io/PaddleSlim/quick_start/quant_aware_tutorial.html)
* 量化训练[API接口说明](https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html)
* 量化训练[Demo](https://github.com/PaddlePaddle/PaddleSlim/tree/release/1.0.1/demo/quant/quant_aware)

## 3 使用Paddle-Lite运行量化模型推理

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


## FAQ

**问题**：Compiled with WITH_GPU, but no GPU found in runtime

**解答**：检查本机是否支持GPU训练，如果不支持请使用CPU训练。如果在docker进行GPU训练，请使用nvidia_docker启动容器。

**问题**：Inufficient GPU memory to allocation. at [/paddle/paddle/fluid/platform/gpu_info.cc:262]
  
**解答**：正确设置run.sh脚本中`CUDA_VISIBLE_DEVICES`，确保显卡剩余内存大于需要内存。
