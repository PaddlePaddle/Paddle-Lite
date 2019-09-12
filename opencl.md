# 基于OpenCL的ARM GPU预测

Lite支持在Android系统上运行基于OpenCL的程序，目前提供armv8和armv7的交叉编译。

## 编译

- 编译环境: 使用基于`paddle/fluid/lite/tools/Dockerfile.mobile`生成docker镜像。
- cmake编译选项介绍
    * `ARM_TARGET_OS` 代表目标操作系统， 目前仅支持android, 亦为默认值。
    * `ARM_TARGET_ARCH_ABI` 代表体系结构类型，支持输入armv8和armv7。其中，armv8等效于arm64-v8a，亦为默认值；armv7等效于 armeabi-v7a。
    * `ARM_TARGET_LANG` 代表编译目标文件所使用的编译器， 默认为gcc，支持 gcc和clang两种。
- 参考示例

```bash
# 假设处于Lite源码根目录下
mkdir -p build_opencl && cd build_opencl
cmake .. \
    -DLITE_WITH_OPENCL=ON \
    -DWITH_GPU=OFF \
    -DWITH_MKL=OFF \
    -DWITH_LITE=ON \
    -DLITE_WITH_CUDA=OFF \
    -DLITE_WITH_X86=OFF \
    -DLITE_WITH_ARM=ON \
    -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
    -DWITH_TESTING=ON \
    -DARM_TARGET_OS="android" -DARM_TARGET_ARCH_ABI="armv8" -DARM_TARGET_LANG="gcc"
# 注意：在编译其他目标对象前，需要先执行如下命令完成OpenCL所需头文件的下载和生成
make opencl_clhpp
# 接着，用户可以选择完整编译
make lite_compile_deps -j4
# 或者选择只编译某一目标文件，例如test_mobilenetv1
make test_mobilenetv1 -j4
```


## 运行示例

- **运行文件准备**

下面以MobileNetV1为例，介绍如何在手机上执行基于OpenCL的ARM GPU推理过程。

**注意：** 以下命令均在Lite源码根目录下运行。

```bash
# 在/data/local/tmp目录下创建OpenCL文件目录
adb shell mkdir -p /data/local/tmp/opencl
adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/buffer
adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/image
# 将OpenCL的kernels文件推送到/data/local/tmp/opencl目录下
adb push lite/opencl/cl_kernel/cl_common.h /data/local/tmp/opencl/cl_kernel/
adb push lite/opencl/cl_kernel/buffer/* /data/local/tmp/opencl/cl_kernel/buffer/
adb push lite/opencl/cl_kernel/image/* /data/local/tmp/opencl/cl_kernel/image/
# 将mobilenet_v1的模型文件推送到/data/local/tmp/opencl目录下
adb shell mkdir -p /data/local/tmp/opencl/mobilenet_v1
adb push build_opencl/third_party/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1/
# 将OpenCL测试程序(如test_mobilenetv1)推送到/data/local/tmp/opencl目录下
adb push build_opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl
```

- **执行OpenCL推理过程**

使用如下命令运行OpenCL程序。其中，`--cl_path`指定了OpenCL的kernels文件即cl\_kernel所在目录，
`--modle_dir`指定了模型文件所在目录。

```bash
adb shell /data/local/tmp/opencl/test_mobilenetv1 --cl_path=/data/local/tmp/opencl --model_dir=/data/local/tmp/opencl/mobilenet_v1 --warmup=1 --repeats=1
```

**注意：** 因为权重参数均会在Op Kernel第一次运行时进行加载，所以第一次的执行时间会略长。一般将warmup的值设为1，repeats值设为多次。


# 如何在Code中使用

Lite支持对ARM CPU和ARM GPU的混调执行，具体描述如下：
1. 设置Lite推断执行的有效Places，使其包含ARM CPU(kARM)和ARM GPU(kOpenCL)；
2. 设置Lite推断执行的偏好Place为ARM GPU(kOpenCL)。

通过以上两步设置，Lite在推断执行过程中如果发现某一Op存在着基于OpenCL的实现，其会优先选择使用该实现执行Op的计算过程。若发现某一Op没有基于OpenCL实现的Kernel，其会自动选择执行基于ARM CPU的实现。

代码示例：
```cpp
DeviceInfo::Init();
DeviceInfo::Global().SetRunMode(LITE_POWER_HIGH, FLAGS_threads);
lite::Predictor predictor;
// 设置Lite推断执行的有效Places为{kHost, kARM, kOpenCL}
std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kOpenCL), PRECISION(kFloat)},
  });
// 设置Lite推断执行的偏好Place为kOpenCL
auto preferred_place = Place({TARGET(kOpenCL), PRECISION(kFloat)});
// 根据有效Places和偏好Place构建模型
predictor.Build(model_dir, preferred_place, valid_places);
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