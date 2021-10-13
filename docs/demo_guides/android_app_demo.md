# Android 工程示例

## 多种应用场景

我们提供的 Paddle Lite 示例工程[ Paddle Lite-Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo)，其中包含[ Android ](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo)、[ iOS ](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-ios-demo)和[ Armlinux ](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-armlinux-demo)平台的示例工程。涵盖[人脸识别](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/face_detection_demo)、[人像分割](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/human_segmentation_demo)、[图像分类](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/image_classification_demo)、[目标检测](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/object_detection_demo)4个应用场景。

### 1. 人脸识别

人脸检测是 Paddle Lite 提供的人像检测 demo 。在移动端上提供了高精度、实时的人脸检测能力，能处理基于人脸检测的业务场景。在移动端预测的效果图如下：

<p align="center"><img width="300" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/face.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="300" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/face2.jpg"/></p>


### 2. 人像分割

人像分割是 Paddle Lite 提供的图像分割 demo ，在移动端上提供了实时的人像分割能力，可以应用证件照自动抠图、面积测量、智能交通（标记车道和交通标志）等场景。  在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/human.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/human2.jpg"/></p>


### 3. 图像分类

图像分类是 Paddle Lite 提供的图像处理 demo ，在移动端上提供了实时的物体识别能力，可以应用到生产线自动分拣或质检、识别医疗图像、辅助医生肉眼诊断等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat2.jpg"/></p>


### 4. 物体检测

物体检测是 Paddle Lite 提供的图像识别 demo ，在移动端上提供了检测多个物体的位置、名称、位置及数量的能力。可以应用到视频监控（是否有违规物体或行为）、工业质检（微小瑕疵的数量和位置）、医疗诊断（细胞计数、中药识别）等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog2.jpg"/></p>


## Android demo部署方法

### 概述

我们推荐你从端侧 Android demo入手，了解 Paddle Lite 应用工程的构建、依赖项配置以及相关API的使用。
本教程基于 [Paddle Lite Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo) 库中的 Android “目标检测示例（ [object_detection_demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/object_detection_demo) )”示例程序，演示端侧部署的流程。

- 选择图像检测模型
- 将模型转换成 Paddle Lite 模型格式，模型转换请见[OPT工具使用文档](../user_guides/model_optimize_tool.md)
- 在端侧使用 Paddle Lite 推理模型

本章将详细说明如何在端侧利用 Paddle Lite  Java API 和 Paddle Lite  图像检测模型完成端侧推理。

### 部署应用

**目的**：将基于 Paddle Lite 预测库的 Android APP 部署到手机，实现物体检测

**需要的环境**： Android Studio、Android 手机（开启 USB 调试模式）、下载到本地的[ Paddle Lite Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo)工程

**预先要求**：如果您的 Android Studio 尚未配置 NDK ，请根据 Android Studio 用户指南中的[安装及配置 NDK 和 CMake ](https://developer.android.com/studio/projects/install-ndk)内容，预先配置好 NDK 。您可以选择最新的 NDK 版本，或者与[ Android 编译环境配置](../source_compile/compile_android.md)中的NDK版本保持一致。

**部署步骤**：

1、目标检测的 Android 示例位于 `Paddle-Lite-Demo\PaddleLite-android-demo\object_detection_demo`

2、用 Android Studio 打开 object_detection_demo 工程 （本步骤需要联网，下载 Paddle Lite 预测库和模型）。

3、手机连接电脑，打开**USB调试**和**文件传输模式**，在 Android Studio 上连接自己的手机设备（手机需要开启允许从 USB 安装软件权限）

![ Android_studio ](https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/Android_studio.png)

>**注意：** 
>> 如果您在导入项目、编译或者运行过程中遇到NDK配置错误的提示，请打开 ` File > Project Structure > SDK Location`，修改 `Andriod NDK location` 为您本机配置的 NDK 所在路径。
>> 如果您是通过 Andriod Studio 的 SDK Tools 下载的 NDK (见本章节"预先要求")，可以直接点击下拉框选择默认路径。
>> 如果以上步骤仍旧无法解决NDK配置错误，请尝试根据 Andriod Studio 官方文档中的[更新 Android Gradle 插件](https://developer.android.com/studio/releases/gradle-plugin?hl=zh-cn#updating-plugin)章节，尝试更新Android Gradle plugin版本。

<p align="center"><img width="600" height="450"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/Andriod_Studio_NDK.png"/></p>

4、按下 Run 按钮，自动编译 APP 并安装到手机。(该过程会自动下载 Paddle Lite 预测库和模型，需要联网)

成功后效果如下，图一：APP 安装到手机        图二： APP 打开后的效果，会自动识别图片中的物体并标记

<p align="center"><img width="300" height="450"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/AndroidApp0.png"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="300" height="450"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/AndroidApp1.jpg"/></p>


## Android demo结构讲解

Android 示例的代码结构如下图所示：

<p align="center"><img width="600" height="450"  src="http://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/Android_struct.png"/></p>


   1、 Predictor.java： 预测代码

```shell
# 位置：
object_detection_demo/app/src/main/java/com/baidu/paddle/lite/demo/object_detection/Predictor.java
```

  2、 model.nb : 模型文件 (opt 工具转化后Paddle Lite模型)；pascalvoc_label_list：训练模型时的`labels`文件

```shell
# 位置：
object_detection_demo/app/src/main/assets/models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb
object_detection_demo/app/src/main/assets/labels/pascalvoc_label_list
# 如果要替换模型，可以将新模型放到`object_detection_demo/app/src/main/assets/models/ssd_mobilenet_v1_pascalvoc_for_cpu`目录下
```

  3、 libpaddle_lite_jni.so、PaddlePredictor.jar：Paddle Lite Java 预测库与Jar包 

```shell
# 位置
object_detection_demo/app/src/main/jniLibs/arm64-v8a/libpaddle_lite_jni.so
object_detection_demo/app/libs/PaddlePredictor.jar
# 如果要替换动态库so和jar文件，则将新的动态库so更新到`object_detection_demo/app/src/main/jniLibs/arm64-v8a/`目录下，新的jar文件更新至`object_detection_demo/app/libs/`目录下
```

  4、 build.gradle : 定义编译过程的 gradle 脚本。（不用改动，定义了自动下载Paddle Lite预测和模型的过程）

```shell
# 位置
object_detection_demo/app/build.gradle
# 如果需要手动更新模型和预测库，则可将grad了脚本中的`download*`接口注释即可
```

## 代码讲解 （使用Paddle Lite Java API 执行预测）

Android 示例基于 Java API 开发，调用 Paddle Lite Java API 包括以下五步。更详细的 API 描述参考： [Paddle Lite Java API](../api_reference/java_api_doc)。

```c++
// 导入Java API
import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.Tensor;
import com.baidu.paddle.lite.Predictor;
import com.baidu.paddle.lite.PowerMode;

// 1. 写入配置：设置MobileConfig
MobileConfig config = new MobileConfig();
config.setModelFromFile(<modelPath>); // 设置Paddle Lite模型路径
config.setPowerMode(PowerMode.LITE_POWER_NO_BIND); // 设置CPU运行模式
config.setThreads(4); // 设置工作线程数

// 2. 创建 PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);

// 3. 设置输入数据
long[] dims = {100, 100};
float[] inputBuffer = new float[10000];
for (int i = 0; i < 10000; ++i) {
    inputBuffer[i] = i;
}
// 如果输入是图片，则可在第三步时将预处理后的图像数据赋值给输入Tensor
Tensor input = predictor.getInput(0);
input.resize(dims);
input.setData(inputBuffer);

// 4. 执行预测
predictor.run();

// 5. 获取输出数据
Tensor result = predictor.getOutput(0);
float[] output = result.getFloatData();
for (int i = 0; i < 1000; ++i) {
    System.out.println(output[i]);
}

// 例如目标检测：输出后处理，输出检测结果
// Fetch output tensor
Tensor outputTensor = getOutput(0);

// Post-process
 long outputShape[] = outputTensor.shape();
 long outputSize = 1;
 for (long s : outputShape) {
   outputSize *= s;
 }
 
 int objectIdx = 0;
 for (int i = 0; i < outputSize; i += 6) {
   float score = outputTensor.getFloatData()[i + 1];
   if (score < scoreThreshold) {
      continue;
   }
   int categoryIdx = (int) outputTensor.getFloatData()[i];
   String categoryName = "Unknown";
   if (wordLabels.size() > 0 && categoryIdx >= 0 && categoryIdx < wordLabels.size()) {
     categoryName = wordLabels.get(categoryIdx);
   }
   float rawLeft = outputTensor.getFloatData()[i + 2];
   float rawTop = outputTensor.getFloatData()[i + 3];
   float rawRight = outputTensor.getFloatData()[i + 4];
   float rawBottom = outputTensor.getFloatData()[i + 5];
   float clampedLeft = Math.max(Math.min(rawLeft, 1.f), 0.f);
   float clampedTop = Math.max(Math.min(rawTop, 1.f), 0.f);
   float clampedRight = Math.max(Math.min(rawRight, 1.f), 0.f);
   float clampedBottom = Math.max(Math.min(rawBottom, 1.f), 0.f);
   // detect_box coordinate 
   float imgLeft = clampedLeft * imgWidth;
   float imgTop = clampedTop * imgWidth;
   float imgRight = clampedRight * imgHeight;
   float imgBottom = clampedBottom * imgHeight;
   objectIdx++;
}

```
## Q&A:
问题：
- 提示某个op not found:
  - 如果编译选项没有打开with_extra的选项，可以打开with_extra的选项再尝试；如果仍存在缺少Op的错误提示，则是目前Paddle-Lite尚未支持该Op，可以在github repo里提issue等待版本迭代，或者参考[添加Op](../develop_guides/add_operation.md)来自行添Op并重新编译。
- 提示in_dims().size() == 4 || in_dims.size() == 5 test error
  - 如果你是基于我们的demo工程替换模型以后出现这个问题，有可能是替换模型以后模型的输入和Paddle-Lite接收的输入不匹配导致，可以参考[这个issue](https://github.com/PaddlePaddle/Paddle-Lite/issues/6406)来解决该问题。
