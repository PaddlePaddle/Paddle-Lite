# Java 完整示例

## 概述

本教程提供了 Paddle Lite 执行推理的示例程序，通过输入、执行推理、打印推理结果的方式，演示了基于 Java API 接口的推理基本流程，用户能够快速了解 Paddle Lite 执行推理相关 API 的使用。

本教程以 Android Studio 工程为案例，介绍 Java API 推理流程，工程文件夹为[lite/demo/java/android](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/java/android)。其中和 Java API 相关的代码在[lite/demo/java/android/PaddlePredictor/app/src/main/java/com/baidu/paddle/lite/MainActivity.java](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/demo/java/android/PaddlePredictor/app/src/main/java/com/baidu/paddle/lite/MainActivity.java)文件中。


使用 Paddle Lite 执行推理主要包括以下步骤：

- 配置 config 信息：创建 MobileConfig ，用于配置模型路径、运行设备环境等相关信息

- 模型加载：通过 `setModelFromFile` 接口配置模型路径。

- 创建 predictor 对象：通过 `PaddlePredictor.createPaddlePredictor` 接口创建 PaddlePredictor 对象，完成模型解析和环境初始化。

- 输入数据：推理之前需要向输入 Tensor 中填充数据。即通过 `predictor.getInput(num)` 接口获取第 `num` 个输入 Tensor ，先做 `resize` 处理，给 Tensor 分配相应的空间；然后通过 `setData` 接口对 Tensor 进行赋值处理。（如果输入数据是图片，则需要进行预处理，再将预处理后的数据赋值给输入 tensor ）

- 执行推理：使用 predictor 对象的成员函数 `run` 进行模型推理

- 输出数据：推理执行结束后，通过 `predictor.getOutput(num)` 接口获取第 `num` 个输出 Tensor。

其流程图如下：


<p align=center> <img src = "https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite/develop/docs/images/predict_workflow.png"/></p>


## Java 应用开发说明

Java 代码调用 Paddle Lite 执行预测仅需五步：

(1) 设置 MobileConfig 信息

```java
MobileConfig config = new MobileConfig();
config.setModelDir(modelPath);
config.setPowerMode(PowerMode.LITE_POWER_HIGH);
config.setThreads(1);
```

(2) 指定模型文件，创建 PaddlePredictor

```java
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);
```

(3) 设置模型输入 (下面以第 i 个输入为 i 为例)

```java
float[] inputBuffer = new float[10000];
for (int i = 0; i < 10000; ++i) {
    inputBuffer[i] = i;
}
Tensor input = predictor.getInput(0);
input.resize({100, 100});
input.setData(inputBuffer);
```

如果模型有多个输入，每一个模型输入都需要准确设置 shape 和 data。

(4) 执行预测

```java
predictor.run();
```

(5) 获得预测结果

```java
Tensor output = predictor.getOutput(0);
```
详细的 Java API 说明文档位于[Java API](../api_reference/java_api_doc)。更多 Java 应用预测开发可以参考位于位于[Paddle Lite Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)的工程示例代码。


## Android Studio 工程 Java 示例程序

本章节展示的所有 Android Studio 工程代码位于 [demo/java](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/java) 。下面将要下载的预测库也已经包含了上述 Android Studio 工程。

### 1. 环境准备

要编译和运行 Android Java 示例程序，你需准备：

1. 一台 armv7 或 armv8 架构的安卓手机
2. 一台装有 Android Studio 的开发机

### 2. 下载预编译的预测库

预测库下载界面位于[Lite预编译库下载](../quick_start/release_lib)，可根据您的手机型号选择合适版本。

以**Android-ARMv8 架构**为例，可以下载以下版本：

| Arch  |with_extra|arm_stl|with_cv|下载|
|:-------:|:-----:|:-----:|:-----:|:-------:|
|armv8|OFF|c++_static|OFF|[2.9-rc](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.9/inference_lite_lib.android.armv8.gcc.c++_static.tar.gz)|

**解压后内容结构如下：**

```shell
inference_lite_lib.android.armv8.gcc.c++_static    Paddle Lite 预测库
├── cxx                                 C++ 预测库
│   ├── include                             C++ 预测库头文件
│   └── lib                                 C++ 预测库文件
├── demo                                示例 Demo
│   ├── cxx                                 C++ 示例 Demo
│   └── java                                Java 示例 Demo
│       ├── README.md                           Java Demo Readme 文件
│       └── android                             Java Andriod Demo
└── java                                Java 预测库
    ├── jar 
    │   └── PaddlePredictor.jar                 Java JAR 包
    ├── so 
    │   └── libpaddle_lite_jni.so               Java JNI 动态链接库
    └── src
```

### 3. 准备预测部署模型

#### 自动化脚本方法

在下载下来的预测库的 `demo/java/android` 文件夹下，为了让您更快上手，我们准备了一个脚本 `prepare_demo.bash`，输入手机架构参数例如 `arm64-v8a`，即可自动准备好所有预测部署所需的文件。

```shell
cd inference_lite_lib.android.armv8.gcc.c++_static/demo/java/android
bash prepare_demo.bash arm8
```

以上命令自动进行以下三步操作：

1. 拷贝 JNI 动态链接库 `libpaddle_lite_jni.so` 到 `PaddlePredictor/app/src/main/jniLibs/arm64-v8a/`
2. 拷贝 JAR 包 `PaddlePredictor.jar` 到 `PaddlePredictor/app/libs/`
3. 自动下载并解压所有模型文件，拷贝到`PaddlePredictor/app/src/main/assets/`

>> **注意：** 目前脚本输入手机架构参数仅支持 `arm7 | arm8 | armeabi-v7a | arm64-v8a`。

#### 手动拷贝方法

如果你不想运行上面的脚本，你可以手动进行下面操作。

(1) 把 Java JNI 动态链接库和 Java JAR 包拷贝进安卓 demo 程序文件夹下：

```shell
cd inference_lite_lib.android.armv8.gcc.c++_static/demo/java/android
# 请替换<架构文件夹>为手机架构名称，例如 arm64-v8a
cp ../../../java/so/libpaddle_lite_jni.so PaddlePredictor/app/src/main/jniLibs/<架构文件夹>
cp ../../../java/jar/PaddlePredictor.jar PaddlePredictor/app/libs/
```

(2) 下载模型文件

下载以下 5 个模型，并解压缩到 `PaddlePredictor/app/src/main/assets` 文件夹中。解压之后，assets文件夹里要包含解压后的五个以`.nb`结尾的模型文件，但不需要保存原压缩`.tar.gz`文件。

| 模型| 下载地址|
| :-- | :-- |
| inception_v4_simple_opt.nb|  http://paddle-inference-dist.bj.bcebos.com/inception_v4_simple_opt.nb.tar.gz |
| lite_naive_model_opt.nb | http://paddle-inference-dist.bj.bcebos.com/lite_naive_model_opt.nb.tar.gz |
| mobilenet_v1_opt.nb | http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1_opt.nb.tar.gz |
| mobilenet_v2_relu_opt.nb|  http://paddle-inference-dist.bj.bcebos.com/mobilenet_v2_relu_opt.nb.tar.gz |
| resnet50_opt.nb| http://paddle-inference-dist.bj.bcebos.com/resnet50_opt.nb.tar.gz |

>> **注意：** 模型要求为 naive buffer 格式，您可以通过 [opt工具](./model_optimize_tool) 将 Paddle 模型转为naive buffer存储格式。

### 4. 运行预测示例程序

1. 用 Android Studio 打开`inference_lite_lib.android.armv8.gcc.c++_static/demo/java/android/PaddlePredictor`文件夹（需联网），打开后工程会自动 build 完成。
2. 设置手机：手机 USB 连接电脑，打开`设置 -> 开发者模式 -> USB调试 -> 允许（授权）当前电脑调试手机`，并确认 Android Studio 可识别接入的手机设备。
3. 按下 Android Studio 的 Run 按钮，Android Studio 会自动编译 APP 并安装到手机。在手机上打开安装成功的 APP ，大概会等 10 秒，然后看到类似以下输出：

```shell
lite_naive_model output: 50.213173, -28.872887
expected: 50.2132, -28.8729

inception_v4_simple test:true
time: xxx ms

resnet50 test:true
time: xxx ms

mobilenet_v1 test:true
time: xxx ms

mobilenet_v2 test:true
time: xxx ms
```

该 demo 程序跑 5 个模型，第一个模型结果将真正的头两个数字输出，并在第二行附上期望的正确值。你应该要看到他们的误差小于 0.001 。后面四个模型如果你看到 `test:true` 字样，说明模型输出通过了我们在 demo 程序里对其输出的测试。time 代表该测试花费的时间。

>> **注意：** 在这一步中，如果遇到 Andriod Studio 编译/安装失败等问题，请参考 [Andriod 示例](../demo_guides/android_app_demo.html#android-demo)中部署方法章节的详细步骤和注意事项。
