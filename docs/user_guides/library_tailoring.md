
# 裁剪预测库

Paddle-Lite支持**根据模型裁剪预测库**功能。Paddle-Lite的一般编译会将所有已注册的operator打包到预测库中，造成库文件体积膨胀；**裁剪预测库**能针对具体的模型，只打包优化后该模型需要的operator，有效降低预测库文件大小。

## 效果展示(Tiny_publish Android动态预测库体积)

| 测试模型 | 裁剪开关  | **libpaddle_lite_jni.so** |转化后模型中的OP|
| ------------------ | ---------------------------- | -------- |------------------|
| mobilenetv1（armv8） | 裁剪前--build_tailor=OFF | 1.5M                | feed,etch,conv2d,depthwise_conv2d,fc,fpool2d,softmax     |
| mobilenetv1（armv8） | 裁剪后--build_tailor=ON  |  788K              |feed,etch,conv2d,depthwise_conv2d,fc,fpool2d,softmax|
| mobilenetv2（armv8） | 裁剪前--build_tailor=OFF  | 1.5M                | feed,fetch,conv2d,depthwise_conv2d,elementwise_add,fc,pool2d,relu6,softmax |
| mobilenetv2（armv8） | 裁剪后--build_tailor=ON  |  912K          |feed,fetch,conv2d,depthwise_conv2d,elementwise_add,fc,pool2d,relu6,softmax|
| mobilenetv1（armv7） | 裁剪前--build_tailor=OFF    | 938K     |feed,fetch,concat,conv2d,dropout,fc,pool2d,softmax|
| mobilenetv1（armv7） | 裁剪后--build_tailor=ON  | 607K   |feed,fetch,concat,conv2d,dropout,fc,pool2d,softmax|
| mobilenetv2（armv7） | 裁剪前--build_tailor=OFF     | 938K | feed,fetch,conv2d,depthwise_conv2d,elementwise_add,fc,pool2d,relu6,softmax |
| mobilenetv2（armv7） | 裁剪后--build_tailor=ON  |687K          |feed,fetch,conv2d,depthwise_conv2d,elementwise_add,fc,pool2d,relu6,softmax|




## 实现过程：


### 1、转化模型时记录优化后模型信息

说明：使用model_optimize_tool转化模型时，选择 `--record_tailoring_info =true`  会将优化后模型的OP和kernel信息保存到输出文件夹，这些信息将用于编译裁剪后的动态库。
注意：需要使用Paddle-Lite 最新版本（release/v2.0.0之后）代码编译出的model_optimize_tool
例如：

```bash
./model_optimize_tool     --model_dir=./mobilenet_v1     --optimize_out_type=naive_buffer     --optimize_out=mobilenet_v1NB     --record_tailoring_info =true     --valid_targets=arm
```
效果：优化后模型使用的OP和kernel信息被保存在 `mobilenet_v1NB`文件夹中的隐藏文件里了

### 2、根据模型信息编译裁剪后的预测库

说明：编译Paddle-Lite时选择`--build_tailor=ON` ，并且用   `–-opt_model_dir=`   指定优化后的模型的地址
例如：

```bash
./lite/tools/build.sh   --arm_os=android   --arm_abi=armv7   --arm_lang=gcc   --android_stl=c++_static   --build_extra=ON --build_tailor=ON --opt_model_dir=../mobilenet_v1NB tiny_publish
```
**注意**：上面命令中的`../mobilenet_v1NB`是第1步得到的转化模型的输出路径

**效果**：编译出来的动态库文件变小，且可以运行优化后的模型。

编译出的C++预测库文件位于  ：

`build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/cxx/lib/`

编译出的Java预测库文件位于：

`build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/java/so/`

### 3、运行裁剪后的预测库文件

注意：基于某一模型裁剪出的预测库只能支持优化工具转化后的该模型，例如根据mobilenetV1裁剪出的 full_api预测库只能运行以protobuf格式转化出的模型mobilenetV1_opt_nb， 裁剪出的light_api预测库只能运行以naive_buffer格式转化出的模型mobilenetV1_opt_nb， 运行其他模型可能会出现`segementation fault:undifined op or kernel`。  模型转化方法参考：[使用opt转化模型](./model_optimize_tool))。



**示例1**：使用裁剪后的light_api预测库运行mobilenetv1

1、执行第二步编译后，light_api的C++ 示例位于

`/Paddle-Lite/build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/demo/cxx/mobile_light`

输入`make`命令执行编译可编译出可执行文件mobilenetv1_light_api

2、使用adb将mobilenetV1_NB模型和mobilenetv1_light_api传到手机后执行demo：

`./mobilenetv1_light_api --model_dir=./mobilenetV1_NB`

注意：`mobilenetV1_NB`是用`mobilenetV1`模型转化的naive_buffer格式模型(不需要设置` --record_tailoring_info =true`，转化流程参考：[使用opt转化模型](./model_optimize_tool))。



**示例2**：使用裁剪后的full_api预测库运行mobilenetv1

1、执行第二步编译后，full_api的C++ 示例位于

`/Paddle-Lite/build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/demo/cxx/mobile_light`

替换mobilenetv1_full_api.cc代码内容：

```C++
#include <gflags/gflags.h>
#include <stdio.h>
#include <vector>
#include "paddle_api.h"          // NOLINT

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

注意：`mobilenetV1_PB`是用`mobilenetV1`模型转化的protobuf格式模型(不需要设置` --record_tailoring_info =true`，转化流程参考：[使用opt转化模型](./model_optimize_tool))。

## 按模型集合裁剪预测库

为了方便用户使用，我们同时提供了按模型集合进行预测库裁剪的功能。用户可以提供一个模型集合，Model Optimize Tool会根据用户所指定的模型集合分析其**优化后的**模型所需要的算子信息对预测库进行裁剪。使用此功能用户根据自己的需要使用模型集合来对预测库中的算子进行任意裁剪。

使用方法如下所示：

```shell
# 非combined模型集合
./model_optimize_tool                     \
    --model_set_dir=<your_model_set_dir>  \
    --optimize_out_type=naive_buffer      \
    --optimize_out=<output_model_set_dir> \
    --record_tailoring_info=true          \
    --valid_targets=arm
   
# combined模型集合
./model_optimize_tool                       \
    --model_set_dir=<your_model_set_dir>    \
    --optimize_out_type=naive_buffer        \
    --model_filename=<model_topo_filename>  \
    --param_filename=<model_param_filename> \
    --optimize_out=<output_model_set_dir>   \
    --record_tailoring_info=true            \
    --valid_targets=arm
```

经过以上步骤后会在`<output_model_set_dir>`中生成模型集合中各模型对应的NaiveBuffer格式的优化模型。此步会对模型集合中所需算子信息进行搜集并存储到`<output_model_set_dir>`中。下一步编译预测库的流程与使用单模型进行预测库裁剪步骤相同。

**注意：**

1. 模型集合**必须**均为combined参数模型或均为非combined参数模型。
2. 使用非combined参数模型时，模型拓扑文件名应为`__model__`，使用非combined参数模型时，集合中各模型的拓扑与参数名应相同，分别由`--model_filename`和`--param_filename`指定。
3. 模型集合**必须**均为INT8量化模型或均为非INT8量化模型。
4. 需要使用Paddle-Lite  `release/v2.1.0`之后版本代码编译出的模型优化工具。
