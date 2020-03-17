# Android Demo

## 多种应用场景

我们提供的Paddle-Lite示例工程[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)，其中包含[Android](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo)、[iOS](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-ios-demo)和[Armlinux](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-armlinux-demo)平台的示例工程。涵盖[人脸识别](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/face_detection_demo)、[人像分割](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/human_segmentation_demo)、[图像分类](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/image_classification_demo)、[目标检测](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/object_detection_demo)4个应用场景。

### 1. 人脸识别

人脸检测是Paddle-Lite提供的人像检测demo。在移动端上提供了高精度、实时的人脸检测能力，能处理基于人脸检测的业务场景。在移动端预测的效果图如下：

<p align="center"><img width="300" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/face.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="300" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/face2.jpg"/></p>

### 2. 人像分割

人像分割是Paddle-Lite 提供的图像分割demo ，在移动端上提供了实时的人像分割能力，可以应用证件照自动抠图、面积测量、智能交通（标记车道和交通标志）等场景。  在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/human.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/human2.jpg"/></p>

### 3. 图像分类

图像分类是Paddle-Lite 提供的图像处理demo ，在移动端上提供了实时的物体识别能力，可以应用到生产线自动分拣或质检、识别医疗图像、辅助医生肉眼诊断等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat2.jpg"/></p>

### 4. 物体检测

物体检测是Paddle-Lite 提供的图像识别demo ，在移动端上提供了检测多个物体的位置、名称、位置及数量的能力。可以应用到视频监控（是否有违规物体或行为）、工业质检（微小瑕疵的数量和位置）、医疗诊断（细胞计数、中药识别）等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog2.jpg"/></p>

## Android demo部署方法

下面我们以 **目标检测示例（object_detection_demo)** 为例讲解如何部署。

**目的**：将基于Paddle-Lite预测库的Android APP 部署到手机，实现物体检测

**需要的环境**： Android Studio、Android手机（开启USB调试模式）、下载到本地的[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)工程

**部署步骤**：

1、 目标检测的Android示例位于 `Paddle-Lite-Demo\PaddleLite-android-demo\object_detection_demo`

2、用Android Studio 打开object_detection_demo工程 （本步骤需要联网）。

3、手机连接电脑，打开**USB调试**和**文件传输模式**，在Android Studio上连接自己的手机设备（手机需要开启允许从 USB安装软件权限）

![Android_studio](https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/Android_studio.png)

4、按下 Run按钮，自动编译APP并安装到手机。(该过程会自动下载Paddle-Lite预测库和模型，需要联网)

成功后效果如下，图一：APP安装到手机        图二： APP打开后的效果，会自动识别图片中的物体并标记

<p align="center"><img width="300" height="450"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/AndroidApp0.png"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="300" height="450"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/AndroidApp1.jpg"/></p>

## Android demo结构讲解

Android 示例的代码结构如下图所示：

<p align="center"><img width="600" height="450"  src="http://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/Android_struct.png"/>


   1、 Predictor.java： 预测代码

```shell
# 位置：
object_detection_demo/app/src/main/java/com/baidu/paddle/lite/demo/object_detection/Predictor.java
```

  2、 model.nb : 模型文件 (opt 工具转化后Paddle-Lite模型)；pascalvoc_label_list：训练模型时的`labels`文件

```shell
# 位置：
object_detection_demo/app/src/main/assets/models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb
object_detection_demo/app/src/main/assets/labels/pascalvoc_label_list
```

  3、 libpaddle_lite_jni.so、PaddlePredictor.jar：Paddle-Lite Java 预测库与Jar包 

```shell
# 位置
object_detection_demo/app/src/main/jniLibs/arm64-v8a/libpaddle_lite_jni.so
object_detection_demo/app/libs/PaddlePredictor.jar
```

  4、 build.gradle : 定义编译过程的 gradle 脚本。（不用改动，定义了自动下载Paddle-Lite预测和模型的过程）

```shell
# 位置
object_detection_demo/app/build.gradle
```



## 代码讲解 （使用Paddle-Lite Java API 执行预测）

Android 示例基于Java API 开发，调用Paddle-Lite Java API包括以下五步。更详细的API 描述参考： [Paddle-Lite Java API](https://paddle-lite.readthedocs.io/zh/latest/api_reference/java_api_doc.html)。

```c++
// 导入Java API
import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.Tensor;
import com.baidu.paddle.lite.Predictor;
import com.baidu.paddle.lite.PowerMode;

// 1. 写入配置：设置MobileConfig
MobileConfig config = new MobileConfig();
config.setModelFromFile(<modelPath>); // 设置Paddle-Lite模型路径
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
```
