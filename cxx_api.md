# C++ API接口使用指南

请参考[源码编译](./source_compile)确保 Lite 可以正确编译，下面用Lite的c++接口加载并执行 MobileNetV1 模型为例，详细说明使用方法。

## 准备模型

Lite支持PaddlePaddle训练好的模型，MobileNetV1模型可以由以下三种方式得到：

- 直接下载训练好的[MobileNetV1模型](https://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz)
- 使用[PaddlePaddle](https://paddlepaddle.org.cn/)构建MobileNetV1网络并训练
- 使用[X2Paddle](./x2paddle)对caffe或者tensorflow的MobileNetV1模型进行转换得到

## 模型优化

使用Model Optimize Tool优化模型，使得模型预测过程表现出优异的性能。Model Optimize Tool的具体使用方法请参考[文档](./model_optimize_tool)。

- 准备model_optimize_tool
- 使用model_optimize_tool优化模型
- 得到优化后的模型，包括__model__.nb文件和param.nb文件

## 加载模型

加载MobileNetV1网络模型，创建predictor，具体可以参考```paddlelite/lite/api/model_test.cc```文件。
```c++
lite::DeviceInfo::Init();
lite::DeviceInfo::Global().SetRunMode(lite::LITE_POWER_HIGH, thread_num);
lite_api::MobileConfig config;
config.set_model_dir(model_dir);

auto predictor = lite_api::CreatePaddlePredictor(config);
```

## 设定输入

得到input_tensor，设置输入值，此处我们设定为全1

```cpp
// 获取第 j 个 tensor 的句柄
auto input_tensor = predictor->GetInput(j);
input_tensor->Resize(input_shapes[j]);

// 获取数据指针，以塞入数据
auto input_data = input_tensor->mutable_data<float>();
int input_num = 1;
for (int i = 0; i < input_shapes[j].size(); ++i) {
  input_num *= input_shapes[j][i];
}
for (int i = 0; i < input_num; ++i) {
  input_data[i] = 1.f;
}
```

## 执行并输出

```cpp
predictor.Run()；
auto* out = predictor.GetOutput(0);
LOG(INFO) << "dims " << out->dims();
LOG(INFO) << "out data size: " << out->data_size();
```

输出为```dims dims{1000,}， out data size: 1000```

