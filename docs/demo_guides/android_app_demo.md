# Android 工程示例

## 多种应用场景

Paddle-Lite 提供了多个应用场景的 Android Demo：
* 图像分类
    * 基于 [mobilenet_v1](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224.tar.gz) 模型 [Android 示例](./image_classification/android/)
     
* 目标检测
    * 基于 [ssd_mobilenetv1](https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/ssd_mobilenet_v1_pascalvoc_fp32_300_fluid.tar.gz) 模型 [Android 示例](./object_detection/android/app/cxx/ssd_mobilenetv1_detection_demo/)

    * 基于 [yolov3_mobilenet_v3](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/yolov3_mobilenet_v3.tar) 模型 [Android 示例](./object_detection/android/app/cxx/yolo_detection_demo/)
     
    * 基于 [yolov5](https://paddlelite-demo.bj.bcebos.com/models/yolov5n/yolov5n.zip) 模型 [Android 示例](./object_detection/android/app/cxx/yolov5n_detection_demo/)

    * 基于 [pp_picodet](https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/picodet_s_320_coco_for_cpu.tar.gz) 模型 [Android 示例](./object_detection/android/app/cxx/picodet_detection_demo/)
    
* 文字识别
    * 基于 [pp_ocr_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_slim_infer.tar)、[pp_ocr_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) 和 [pp_ocr_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_cls_slim_infer.tar) 模型 [Android 示例](./ocr/android/)
    
* 人脸检测
    * 基于 [face-detection](https://paddlelite-demo.bj.bcebos.com/models/facedetection_fp32_240_430_fluid.tar.gz) 模型 [Android 示例](./face_detection/android/)
     
* 人脸关键点检测
    * 基于 [face-detection](https://paddlelite-demo.bj.bcebos.com/models/facedetection_fp32_240_430_fluid.tar.gz) 和 [face-keypoint](https://paddlelite-demo.bj.bcebos.com/models/facekeypoints_detector_fp32_60_60_fluid.tar.gz) 模型 [Android 示例](./face_keypoints_detection/android/)
    
* 口罩识别
    * 基于 [pyramidbox](https://paddlelite-demo.bj.bcebos.com/models/pyramidbox_lite_fp32_fluid.tar.gz) + [mask_detect](https://paddlelite-demo.bj.bcebos.com/models/mask_detector_fp32_128_128_fluid.tar.gz) 模型 [Android 示例](./mask_detection/android/)
    
* 人像分割
    * 基于 [DeeplabV3](https://paddlelite-demo.bj.bcebos.com/models/deeplab_mobilenet_fp32_fluid.tar.gz) 模型 [Android 示例](./human_segmentation/android/)

* PP 识图
   * 基于 [PPLCNet](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/lite/ppshitu_lite_models_v1.0.tar) 两个模型模型 [Android 示例](./PP_shitu/android/)


### 1. 人脸识别

人脸检测是 Paddle Lite 提供的人像检测 Demo ，在移动端上提供了高精度、实时的人脸检测能力，能处理基于人脸检测的业务场景。在移动端预测的效果图如下：

<p align="center"><img width="300" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/face.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="300" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/face2.jpg"/></p>


### 2. 人像分割

人像分割是 Paddle Lite 提供的图像分割 Demo ，在移动端上提供了实时的人像分割能力，可以应用证件照自动抠图、面积测量、智能交通（标记车道和交通标志）等场景。  在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/human.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/human2.jpg"/></p>


### 3. 图像分类

图像分类是 Paddle Lite 提供的图像处理 Demo ，在移动端上提供了实时的物体识别能力，可以应用到生产线自动分拣或质检、识别医疗图像、辅助医生肉眼诊断等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/tabby_cat2.jpg"/></p>


### 4. 物体检测

物体检测是 Paddle Lite 提供的图像识别 Demo ，在移动端上提供了检测多个物体的位置、名称、位置及数量的能力。可以应用到视频监控（是否有违规物体或行为）、工业质检（微小瑕疵的数量和位置）、医疗诊断（细胞计数、中药识别）等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/demo/dog2.jpg"/></p>

### 5. 文字识别

文字识别是 Paddle Lite 提供的OCR类文字识别 Demo ，在移动端上提供了检测多行文字的位置和名称的能力。可以应用到中英文翻译、词典笔等场景。在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/android/test.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="250" height="250"  src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/ios/run_app.jpeg"/></p>

### 6. PP 识图

PP 识图是 Paddle Lite 提供的识别图片内容和位置 Demo ，在移动端上提供了检测多个物体的位置和名称的能力，在移动端预测的效果图如下：

<p align="center"><img width="250" height="250"  src="https://paddlelite-demo.bj.bcebos.com/demo/PP_shitu/doc_img/wu_ling.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<p align="center"><img width="250" height="250"  src="https://paddlelite-demo.bj.bcebos.com/demo/PP_shitu/doc_img/app_interface.jpg"/></p>

## Android demo部署方法

### 概述

我们推荐你从端侧 Android demo 入手，了解 Paddle Lite 应用工程的构建、依赖项配置以及相关 `API` 的使用。
本教程基于[ Paddle Lite Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo) 库中的 Android “目标检测示例（[ object_detection_demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo/object_detection_demo) )”示例程序，演示端侧部署的流程。

- 选择图像检测模型
- 将模型转换成 Paddle Lite 模型格式，模型转换请见[ OPT 工具使用文档](../user_guides/model_optimize_tool.md)
- 在端侧使用 Paddle Lite 推理模型

本章将详细说明如何在端侧利用 Paddle Lite  `Java API` 和 Paddle Lite  图像检测模型完成端侧推理。

### 部署应用

**目的**：将基于 Paddle Lite 预测库的 Android APP 部署到手机，实现物体检测

**需要的环境**： Android Studio、Android 手机（开启 USB 调试模式）、下载到本地的[ Paddle Lite Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo)工程

**预先要求**：如果您的 Android Studio 尚未配置 NDK ，请根据 Android Studio 用户指南中的[安装及配置 NDK 和 CMake ](https://developer.android.com/studio/projects/install-ndk)内容，预先配置好 NDK 。您可以选择最新的 NDK 版本，或者与[ Linux x86 环境下编译适用于 Android 的库](../source_compile/linux_x86_compile_android)、[macOS 环境下编译适用于 Android 的库](../source_compile/macos_compile_android) 两个章节中的 NDK 版本保持一致。

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


## Android demo 结构讲解

Android 示例的代码结构如下图所示：

<p align="center"><img width="600" height="450"  src="http://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/android/Android_struct.png"/></p>


   1、 `Predictor.java`： 预测代码

```shell
# 位置：
object_detection_demo/app/src/main/java/com/baidu/paddle/lite/demo/object_detection/Predictor.java
```

  2、 `model.nb` : 模型文件 (opt 工具转化后 Paddle Lite 模型), `pascalvoc_label_list`：训练模型时的 `labels` 文件

```shell
# 位置：
object_detection_demo/app/src/main/assets/models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb
object_detection_demo/app/src/main/assets/labels/pascalvoc_label_list
# 如果要替换模型，可以将新模型放到 `object_detection_demo/app/src/main/assets/models/ssd_mobilenet_v1_pascalvoc_for_cpu` 目录下
```

  3、 `libpaddle_lite_jni.so、PaddlePredictor.jar`：Paddle Lite Java 预测库与 Jar 包 

```shell
# 位置
object_detection_demo/app/src/main/jniLibs/arm64-v8a/libpaddle_lite_jni.so
object_detection_demo/app/libs/PaddlePredictor.jar
# 如果要替换动态库 so 和 jar 文件，则将新的动态库 so 更新到 `object_detection_demo/app/src/main/jniLibs/arm64-v8a/` 目录下，新的 jar 文件更新至 `object_detection_demo/app/libs/` 目录下
```

  4、`build.gradle` : 定义编译过程的 gradle 脚本。（不用改动，定义了自动下载 Paddle Lite 预测和模型的过程）

```shell
# 位置
object_detection_demo/app/build.gradle
# 如果需要手动更新模型和预测库，则可将 gradle 脚本中的 `download*` 接口注释即可
```

## 代码讲解 （使用 Paddle Lite `Java API` 执行预测）

Android 示例基于 Java API 开发，调用 Paddle Lite `Java API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite Java API ](../api_reference/java_api_doc)。

```c++
// 导入 Java API
import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.Tensor;
import com.baidu.paddle.lite.Predictor;
import com.baidu.paddle.lite.PowerMode;

// 1. 写入配置：设置 MobileConfig
MobileConfig config = new MobileConfig();
config.setModelFromFile(<modelPath>); // 设置 Paddle Lite 模型路径
config.setPowerMode(PowerMode.LITE_POWER_NO_BIND); // 设置 CPU 运行模式
config.setThreads(4); // 设置工作线程数

// 2. 创建 PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);

// 3. 设置输入数据
long[] dims = {100, 100};
float[] inputBuffer = new float[10000];
for (int i = 0; i < 10000; ++i) {
    inputBuffer[i] = i;
}
// 如果输入是图片，则可在第三步时将预处理后的图像数据赋值给输入 Tensor
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

### Q&A:
问题：
- 提示某个 `op not found`:
  - 如果编译选项没有打开 `with_extra` 的选项，可以打开 `with_extra` 的选项再尝试；如果仍存在缺少 `op` 的错误提示，则是目前 Paddle Lite 尚未支持该 `op` ，可以在 github repo 里提 issue 等待版本迭代，或者参考[添加 op ](../develop_guides/add_operation.md)来自行添 `op` 并重新编译。
- 提示 `in_dims().size() == 4 || in_dims.size() == 5 test error`
  - 如果你是基于我们的 demo 工程替换模型以后出现这个问题，有可能是替换模型以后模型的输入和 Paddle Lite 接收的输入不匹配导致，可以参考[ issue 6406 ](https://github.com/PaddlePaddle/Paddle-Lite/issues/6406)来解决该问题。
- 如果想进一步提高 APP 速度：
  - 可以将 APP 的默认线程数由线程数 1 更新为多线程，如 2/4 线程。另外，APP 的 setting 界面提供了多线程选项，即可在 setting 界面进行线程数更新，不用重新编译和安装啦。
  - 多线程使用限制：线程数最大值是手机大核处理器的个数，如小米 9，它由 4 个 A76 大核组成，即最大运行 4 个线程。
  - 多线程预测库：GCC 编译，V7/V8 多线程均支持；clang 编译下，只支持V8 多线程，V7 多线程编译受限于 NDK，当前 NDK >= 17, 编译报错，问题来源 NDK 内部 clang 编译的寄存器数目限制。
- 如果想用 FP16 模型推理：
  - 更新预测库：包含FP16 kernel的预测库，可以在 [release 官网](https://github.com/PaddlePaddle/Paddle-Lite/tags)下载，也可以参考[源码编译文档](../source_compile/macos_compile_android.rst)，自行编译。
  - 更新 nb 模型：需要使用 OPT 工具，将 `enable_fp16` 设置为 ON，重新转换模型。
  - FP16 预测库和 FP16 模型只在**V8.2 架构以上的手机**上运行，即高端手机，如小米 9，华为 P30 等
