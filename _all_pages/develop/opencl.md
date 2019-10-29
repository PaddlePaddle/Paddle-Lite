---
layout: post
title: 基于OpenCL的ARM GPU预测
---

Lite支持在Android系统上运行基于OpenCL的程序，目前支持Ubuntu环境下armv8、armv7的交叉编译。

## 编译

编译环境：

1. Docker 容器环境；
2. Linux（推荐 Ubuntu 16.04）环境。

详见[ **源码编译指南-环境准备** 章节](./source_compile.md)

编译选项：

|参数|介绍|值|
|--------|--------|--------|
|--arm_os|代表目标操作系统|目前仅支持且默认为`android`|
|--arm_abi|代表体系结构类型，支持armv8和armv7|默认为`armv8`即arm64-v8a；`armv7`即armeabi-v7a|
|--arm_lang|代表编译目标文件所使用的编译器|默认为gcc，支持 gcc和clang两种|

编译范例（以Docker容器环境为例，CMake3.10，android-ndk-r17c位于`/opt/目录下`）：

```bash
# 假设当前位于处于Lite源码根目录下

# 导入NDK_ROOT变量，注意检查您的安装目录若与本示例不同
export NDK_ROOT=/opt/android-ndk-r17c

# 删除上一次CMake自动生成的.h文件
rm ./lite/api/paddle_use_kernels.h
rm ./lite/api/paddle_use_ops.h

# 根据指定编译参数编译
./lite/tools/ci_build.sh \
  --arm_os=android \
  --arm_abi=armv8 \
  --arm_lang=gcc \
  build_test_arm_opencl
```


## 运行示例

- **运行文件准备**

下面以android、ARMv8、gcc的环境为例，介绍如何在手机上执行基于OpenCL的ARM GPU推理过程。

**注意：** 以下命令均在Lite源码根目录下运行。

```bash
# 在/data/local/tmp目录下创建OpenCL文件目录
adb shell mkdir -p /data/local/tmp/opencl
adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/buffer
adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/image

# 将OpenCL的kernels文件推送到/data/local/tmp/opencl目录下
adb push lite/backends/opencl/cl_kernel/cl_common.h /data/local/tmp/opencl/cl_kernel/
adb push lite/backends/opencl/cl_kernel/buffer/* /data/local/tmp/opencl/cl_kernel/buffer/
adb push lite/backends/opencl/cl_kernel/image/* /data/local/tmp/opencl/cl_kernel/image/

# 将mobilenet_v1的模型文件推送到/data/local/tmp/opencl目录下
adb shell mkdir -p /data/local/tmp/opencl/mobilenet_v1
adb push build.lite.android.armv8.gcc.opencl/third_party/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1/
# 将OpenCL测试程序(如test_mobilenetv1)推送到/data/local/tmp/opencl目录下
adb push build.lite.android.armv8.gcc.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl
```

- **执行OpenCL推理过程**

使用如下命令运行OpenCL程序。其中，`--cl_path`指定了OpenCL的kernels文件即cl\_kernel所在目录，
`--modle_dir`指定了模型文件所在目录。

```bash
adb shell chmod +x /data/local/tmp/opencl/test_mobilenetv1
adb shell /data/local/tmp/opencl/test_mobilenetv1 \
  --cl_path=/data/local/tmp/opencl \
  --model_dir=/data/local/tmp/opencl/mobilenet_v1 \
  --warmup=1 \
  --repeats=1
```

**注意：** 因为权重参数均会在Op Kernel第一次运行时进行加载，所以第一次的执行时间会略长。一般将warmup的值设为1，repeats值设为多次。


# 如何在Code中使用

Lite支持对ARM CPU和ARM GPU的混调执行，具体描述如下：

- 设置Lite推断执行的有效Places，使其包含ARM CPU(kARM)和ARM GPU(kOpenCL)；
- 确保GPU(kOpenCL)在第一位，位置代表Places的重要性和kernel选择有直接关系。  
G
通过以上设置，Lite在推断执行过程中如果发现某一Op存在着基于OpenCL的实现，其会优先选择使用该实现执行Op的计算过程。若发现某一Op没有基于OpenCL实现的Kernel，其会自动选择执行基于ARM CPU的实现。

代码示例：
```cpp
DeviceInfo::Init();
DeviceInfo::Global().SetRunMode(LITE_POWER_HIGH, FLAGS_threads);
lite::Predictor predictor;

// 设置Lite推断执行的硬件信息Places为{kOpenCL, kARM}
std::vector<Place> valid_places({
      Place({TARGET(kOpenCL), PRECISION(kFloat)}),
      Place({TARGET(kARM), PRECISION(kFloat)})
  });

// 根据Place构建模型
predictor.Build(model_dir, "", "", valid_places);

// 设置模型的输入
auto* input_tensor = predictor.GetInput(0);
input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
auto* data = input_tensor->mutable_data<float>();
auto item_size = input_tensor->dims().production();
for (int i = 0; i < item_size; i++) {
  data[i] = 1;
}

// 执行模型推断并获取模型的预测结果
predictor.Run();
auto* out = predictor.GetOutput(0);
```

# 其它注意

因OpenCL有两种形式：cl::Image2D和cl::Buffer，如果出现 segmentationFault 的情况，很有可能是因为OpenCL在选择kernel的时候，上一个kernel的输出是cl::Buffer或者cl::Image2D的格式，下一个kernel的输入是cl::Image2D或cl::Buffer，导致不匹配出现的问题。
