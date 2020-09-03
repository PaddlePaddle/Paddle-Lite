# Java 完整示例

本章节展示的所有Java 示例代码位于 [demo/java](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/java) 。

## 1. 环境准备

要编译和运行Android Java 示例程序，你需要准备：

1. 一台armv7或armv8架构的安卓手机
2. 一台装有AndroidStudio的开发机

## 2. 下载预编译的预测库

预测库下载界面位于[Paddle-Lite官方预编译库](release_lib)，可根据您的手机型号选择合适版本。

以**Android-ARMv8架构**为例，可以下载以下版本：

| Arch  |with_extra|arm_stl|with_cv|下载|
|:-------:|:-----:|:-----:|:-----:|:-------:|
|armv8|OFF|c++_static|OFF|[release/v2.6.1](https://paddlelite-data.bj.bcebos.com/Release/2.6.1/Android/inference_lite_lib.android.armv8.gcc.c++_static.CV_OFF.tar.gz)|

**解压后内容结构如下：**

```shell
inference_lite_lib.android.armv8          # Paddle-Lite 预测库
├── cxx                                       # C++ 预测库
│   ├── include                                   # C++ 预测库头文件
│   └── lib                                       # C++ 预测库文件
├── demo                                      # 示例 Demo
│   ├── cxx                                       # C++ 示例 Demo
│   └── java                                      # Java 示例 Demo
│       ├── README.md                                 # Demo Readme 文件
│       └── android                                   # Java Andriod Demo
└── java                                      # Java 预测库
    ├── jar 
    │   └── PaddlePredictor.jar                    # Java JAR 包
    ├── so 
    │   └── libpaddle_lite_jni.so                  # Java JNI 动态链接库
    └── src
```

## 3. 准备预测部署模型

### 自动化脚本方法

在Java Andriod Demo文件夹下，我们准备了一个脚本`prepare_demo.bash`，输入手机架构参数例如`arm64-v8a`，即可自动打包所有预测部署所需文件。

```
cd inference_lite_lib.android.armv8/demo/java/android
bash prepare_demo.bash arm8
```

以上命令自动进行了以下三步操作：

1. 拷贝JNI动态链接库`libpaddle_lite_jni.so`到`PaddlePredictor/app/src/main/jniLibs/arm64-v8a/`
2. 拷贝JAR包`PaddlePredictor.jar` 到 `PaddlePredictor/app/libs/`
3. 自动下载并解压所有模型文件，拷贝到`PaddlePredictor/app/src/main/assets/`

**注意：** 目前脚本输入手机架构参数仅支持 `arm7 | arm8 | armeabi-v7a | arm64-v8a`。

### 手动拷贝方法

(1) 把Java JNI动态链接库和Java JAR包拷贝进安卓demo程序文件夹下：

```shell
cd inference_lite_lib.android.armv8/demo/java/android
# 请替换<架构文件夹>为手机架构名称，例如 arm64-v8a
cp ../../../java/so/libpaddle_lite_jni.so PaddlePredictor/app/src/main/jniLibs/<架构文件夹>
cp ../../../java/jar/PaddlePredictor.jar PaddlePredictor/app/libs/
```

(2) 下载模型文件

下载以下5个模型，并解压缩到 `PaddlePredictor/app/src/main/assets` 文件夹中。解压之后，assets文件夹里要包含解压后的五个以`.nb`结尾的模型文件，但不需要保存原压缩`.tar.gz`文件。

| 模型| 下载地址|
| :-- | :-- |
| inception_v4_simple_opt.nb|  http://paddle-inference-dist.bj.bcebos.com/inception_v4_simple_opt.nb.tar.gz |
| lite_naive_model_opt.nb | http://paddle-inference-dist.bj.bcebos.com/lite_naive_model_opt.nb.tar.gz |
| mobilenet_v1_opt.nb | http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1_opt.nb.tar.gz |
| mobilenet_v2_relu_opt.nb|  http://paddle-inference-dist.bj.bcebos.com/mobilenet_v2_relu_opt.nb.tar.gz |
| resnet50_opt.nb| http://paddle-inference-dist.bj.bcebos.com/resnet50_opt.nb.tar.gz |

注意：模型要求为naive buffer格式，您可以通过 [opt工具](../user_guides/model_optimize_tool) 将Paddle模型转为naive buffer存储格式。

## 4. 运行预测示例程序

1. 用AndroidStudio打开`inference_lite_lib.android.armv8/demo/java/android/PaddlePredictor`文件夹（需要联网），打开后工程会自动build完成。
2. 设置手机：手机USB连接电脑，打开`设置 -> 开发者模式 -> USB调试 -> 允许（授权）当前电脑调试手机`，并确认AndroidStudio可以识别接入的手机设备。
3. 按下AndroidStudio的Run按钮，AndroidStudio会自动编译APP并安装到手机。在手机上打开安装成功的APP，大概会等10秒，然后看到类似以下输出：

```
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

该 demo 程序跑我们的 5 个模型，第一个模型结果将真正的头两个数字输出，并在第二行附上期望的正确值。你应该要看到他们的误差小于0.001。后面四个模型如果你看到 `test:true` 字样，说明模型输出通过了我们在 demo 程序里对其输出的测试。time 代表该测试花费的时间。

**注意：** 在这一步中，如果遇到Andriod Studio编译/安装失败等问题，请参考[Andriod示例](../demo_guides/android_app_demo.html#id6)中部署方法章节的详细步骤和注意事项。
