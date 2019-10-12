---
layout: post
title: Lite支持NPU在线编译
---

Paddle Lite可以在线分析模型特点，在线编译并生成NPU所需要的IR并实时运行。
是首个支持NPU在线模型的预测框架。

也可以离线分析并调优模型后，保存离线模型，直接线上部署使用。

# 编译

只需要提前准备华为DKK库和Lite 代码。

我们也提供了编译NPU的[脚本](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tools/build_npu.sh)可以直接使用。

例如：
```shell
$ ./lite/tools/build_npu.sh --arm_os=android --arm_abi=armv8 --arm_lang=gcc --ddk_root=/to/your/ddk_path build
```

## 细节说明

CMAKE编译选项：

- 设置`LITE_WITH_NPU=ON`和`LITE_WITH_ARM=ON`
- 设置DDK根目录路径 `NPU_DDK_ROOT`

其他编译选项与ARM编译相同，可以参考[“Paddle Lite在Docker下的ARM编译”](../source_compile)。

示例如下：
```shell
    cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DWITH_ARM_DOTPROD=ON   \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=ON \
        -DLITE_WITH_NPU=ON \
        -DANDROID_API_LEVEL=24 \
        -DNPU_DDK_ROOT="/path/to/ai_ddk_lib/" \
        -DARM_TARGET_OS=android -DARM_TARGET_ARCH_ABI=armv8 -DARM_TARGET_LANG=gcc
    make test_mobilenetv1 -j
```

Note： 当前仅支持armv8和gcc编译。

# 运行示例

把MobilenetV1的模型和参数push到指定的`working_dir`.

```shell

working_dir=/data/local/tmp

test_bin=test_npu_pass
model_dir=mobilenet_v1 # as example
repeats=10
batch_size=1
im_channel=3
im_height=224
im_width=224
optimized_model="${model_dir}_opt"

adb shell "mkdir -p ${working_dir}"
adb push $test_bin $working_dir/
adb push $model_dir $working_dir
adb push ai_ddk_lib/lib64/* $working_dir
adb shell chmod +x "${working_dir}/${test_bin}"
adb shell "rm -rf ${working_dir}/${optimized_model}"

adb shell "cd ${working_dir} ; export LD_LIBRARY_PATH=./; ./${test_bin} --model_dir=${model_dir} --optimized_model=${optimized_model} --repeats=${repeats} --batch_size=${batch_size} --im_channel=${im_channel} --im_height=${im_height} --im_width=${im_width}"

```
在华为810的机器上，由运行结果可知单侧通过并且预测速度为6ms左右。
一般第一次的运行时间略长，可以重复多次得到稳定结果。

# 如何在Code中使用

在Lite中使用NPU非常简单，不需要添加太多额外代码。

- 只需要在添加有效place的时候包括`Place{TARGET(kNPU), PRECISION(kFloat)}`即可。
后续的运行和使用没有任何变化。

Note：
唯一需要注意的是，因为构建NPU子图需要提前知晓各个op输入的具体大小，所以生成NPU的`RuntimeProgram`时需要提前初始化输入的大小，主要包括batchsize大小。
如果不提前设置好大小，生成NPU模型时会报错退出。

代码示例：
```cpp
std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                   Place{TARGET(kARM), PRECISION(kFloat)}});
// if want to use NPU
valid_places.push_back(Place{TARGET(kNPU), PRECISION(kFloat)});

DeviceInfo::Init();
DeviceInfo::Global().SetRunMode(LITE_POWER_HIGH, FLAGS_threads);
lite::Predictor predictor;
predictor.Build(model_dir, preferred_place, valid_places);

auto* input_tensor = predictor.GetInput(0);
input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
auto* data = input_tensor->mutable_data<float>();
auto item_size = input_tensor->dims().production();
for (int i = 0; i < item_size; i++) {
  data[i] = 1;
}

predictor.Run();
```

# FAQ

## 关于开发板

由于该框架针对的是华为HiAI最新的NPU架构，应该还没有现成的开发板集成了该架构的NPU，所以通常看到的比如海思2359A上的NPU不一样的。

## 关于手机

支持目前最新的是华为810，以及未来要发布的NPU系列手机。

# Note

注意：由于我们的开发是基于华为内部的最新DDK版本编译，如果您的DDK不是最新的，有可能会遇到编译时某个op找不到定义的情况，此时您可以联系我们尝试一起解决。
