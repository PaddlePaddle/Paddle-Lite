---
layout: post
title: 裁剪预测库方法
---
* TOC
{:toc}

## 裁剪预测库方法

Paddle-Lite支持**根据模型裁剪预测库**功能。Paddle-Lite的一般编译会将所有已注册的operator打包到预测库中，造成库文件体积膨胀；**裁剪预测库**能针对具体的模型，只打包优化后该模型需要的operator，有效降低预测库文件大小。

## 效果展示

| mobilenet_v1 | libpaddle_full_api_shared.so | libpaddle_light_api_shared.so | libpaddle_lite_jni.so |
| ------------------ | ---------------------------- | ----------------------------- | --------------------- |
| mobilenet_v1       | 14M                          | 14M                           | 6.2M                  |
| 裁剪后mobilenet_v1 | 7.7M                         | 7.5M                          | 2.5M                  |

## 实现过程：


### 1、转化模型时记录优化后模型信息

说明：使用model_optimize_tool转化模型时，选择 `--record_tailoring_info =true`  会将优化后模型的OP和kernel信息保存到输出文件夹，这些信息将用于编译裁剪后的动态库。
注意：需要使用Paddle-Lite 最新版本（release/v2.0.0之后）代码编译出的model_optimize_tool
例如：

```
./model_optimize_tool     --model_dir=./mobilenet_v1     --optimize_out_type=naive_buffer     --optimize_out=mobilenet_v1NB     --record_tailoring_info =true     --valid_targets=arm
```
效果：优化后模型使用的OP和kernel信息被保存在 `mobilenet_v1NB`文件夹中的隐藏文件里了

### 2、根据模型信息编译裁剪后的预测库

说明：编译Paddle-Lite时选择`--build_tailor=ON` ，并且用   `–-opt_model_dir=`   指定优化后的模型的地址
例如：

```
./lite/tools/build.sh   --arm_os=android   --arm_abi=armv7   --arm_lang=gcc   --android_stl=c++_static   --build_extra=ON --build_tailor=ON --opt_model_dir=../mobilenet_v1NB full_publish
```
**注意**：上面命令中的`../mobilenet_v1NB`是第1步得到的转化模型的输出路径

**效果**：编译出来的动态库文件变小，且可以运行优化后的模型。

|                    | libpaddle_full_api_shared.so | libpaddle_light_api_shared.so | libpaddle_lite_jni.so |
| ------------------ | ---------------------------- | ----------------------------- | --------------------- |
| mobilenet_v1       | 14M                          | 14M                           | 6.2M                  |
| 裁剪后mobilenet_v1 | 7.7M                         | 7.5M                          | 2.5M                  |

编译出的C++预测库文件位于  ：

`build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/cxx/lib/`

编译出的Java预测库文件位于：

`build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/java/so/`

### 3、运行裁剪后的预测库文件

注意：基于某一模型裁剪出的预测库只能支持优化工具转化后的该模型，例如根据mobilenetV1裁剪出的 full_api预测库只能运行以protobuf格式转化出的模型mobilenetV1_opt_nb， 裁剪出的light_api预测库只能运行以naive_buffer格式转化出的模型mobilenetV1_opt_nb， 运行其他模型可能会出现`segementation fault:undifined op or kernel`。  模型转化方法参考：[使用model_optimize_tool转化模型](../model_optimize_tool))。



**示例1**：使用裁剪后的light_api预测库运行mobilenetv1

1、执行第二步编译后，light_api的C++ 示例位于

`/Paddle-Lite/build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/demo/cxx/mobile_light`

输入`make`命令执行编译可编译出可执行文件mobilenetv1_light_api

2、使用adb将mobilenetV1_NB模型和mobilenetv1_light_api传到手机后执行demo：

`./mobilenetv1_light_api --model_dir=./mobilenetV1_NB`

注意：`mobilenetV1_NB`是用`mobilenetV1`模型转化的naive_buffer格式模型(不需要设置` --record_tailoring_info =true`，转化流程参考：[使用model_optimize_tool转化模型](../model_optimize_tool))。



**示例2**：使用裁剪后的full_api预测库运行mobilenetv1
1、执行第二步编译后，full_api的C++ 示例位于

`/Paddle-Lite/build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/demo/cxx/mobile_light`

替换mobilenetv1_full_api.cc代码内容：

```C++
#include <gflags/gflags.h>
#include <stdio.h>
#include <vector>
#include "paddle_api.h"          // NOLINT
#include "paddle_use_kernels.h"  // NOLINT
#include "paddle_use_ops.h"      // NOLINT
#include "paddle_use_passes.h"   // NOLINT

using namespace paddle::lite_api;  // NOLINT

DEFINE_string(model_dir, "", "Model dir path.");

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

void RunModel() {
  // 1. Set CxxConfig
  CxxConfig config;
  config.set_model_file(FLAGS_model_dir + "model");
  config.set_param_file(FLAGS_model_dir + "params");

  std::vector<Place> valid_places{Place{TARGET(kARM), PRECISION(kFloat)}};
  config.set_valid_places(valid_places);

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<CxxConfig>(config);

  // 3. Prepare input data
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize(shape_t({1, 3, 224, 224}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
    data[i] = 1;
  }

  // 4. Run predictor
  predictor->Run();

  // 5. Get output
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  printf("Output dim: %d\n", output_tensor->shape()[1]);
  for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
    printf("Output[%d]: %f\n", i, output_tensor->data<float>()[i]);
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  RunModel();
  return 0;
}

```

2、使用adb将mobilenetV1_PB模型和mobilenetv1_full_api传到手机后执行demo：

`./mobilenetv1_full_api --model_dir=./mobilenetV1_PB`

注意：`mobilenetV1_PB`是用`mobilenetV1`模型转化的protobuf格式模型(不需要设置` --record_tailoring_info =true`，转化流程参考：[使用model_optimize_tool转化模型](../model_optimize_tool))。


