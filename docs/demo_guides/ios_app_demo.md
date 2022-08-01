# iOS 工程示例

## 多种应用场景

Paddle-Lite 提供了多个应用场景的 iOS Demo:
* 图像分类
    * 基于 [mobilenet_v1](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224.tar.gz) 模型 [iOS 示例](./image_classification/ios/)    
* 目标检测
    * 基于 [ssd_mobilenetv1](https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/ssd_mobilenet_v1_pascalvoc_fp32_300_fluid.tar.gz) 模型 [iOS 示例](./object_detection/ios/ssd_mobilenetv1_demo/)
    * 基于 [yolov3_mobilenet_v3](https://paddlemodels.bj.bcebos.com/object_detection/mobile_models/lite/yolov3_mobilenet_v3.tar) 模型 [iOS 示例](./object_detection/ios/yolov3_mobilenet_v3_demo/)
  
    * 基于 [pp_picodet](https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/picodet_s_320_coco_for_cpu.tar.gz) 模型 [iOS 示例](./object_detection/ios/picodet_demo/)
* 文字识别
    * 基于 [pp_ocr_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_slim_infer.tar)、[pp_ocr_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) 和 [pp_ocr_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_cls_slim_infer.tar) 模型 [iOS 示例](./ocr/ios/)
* 人脸检测
    * 基于 [face-detection](https://paddlelite-demo.bj.bcebos.com/models/facedetection_fp32_240_430_fluid.tar.gz) 模型 [iOS 示例]((./face_detection/ios/face_detection))
* 人脸关键点检测
    * 基于 [face-detection](https://paddlelite-demo.bj.bcebos.com/models/facedetection_fp32_240_430_fluid.tar.gz) 和 [face-keypoint](https://paddlelite-demo.bj.bcebos.com/models/facekeypoints_detector_fp32_60_60_fluid.tar.gz) 模型 [iOS 示例](./face_keypoints_detection/ios/face_keypoints_detection)
* 人像分割
    * 基于 [DeeplabV3](https://paddlelite-demo.bj.bcebos.com/models/deeplab_mobilenet_fp32_fluid.tar.gz) 模型 [iOS 示例](./human_segmentation/ios/human_segmentation)

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

## IOS Demo 部署方法

下面我们以**目标检测( picodet_demo )** 为例讲解如何部署 IOS 工程。

**目的**：将基于 Paddle Lite 预测库的 IOS APP部署到苹果手机，实现物体检测。

**需要的环境**：Mac 电脑上安装 Xcode、苹果手机、下载到本地的[ Paddle-Lite-Demo ](https://github.com/PaddlePaddle/Paddle-Lite-Demo)工程

**部署步骤**：

1、 目标检测 Demo 位于 Paddle-Lite-Demo/object_detection/ios/ssd_mobilnetv1_demo 目录
2、`cd Paddle-Lite-Demo/libs` 目录，运行 `download.sh` 脚本，下载所需要的 Paddle Lite 预测库
3、`cd Paddle-Lite-Demo/object_detection/assets` 目录，运行 `download.sh` 脚本，下载 OPT 优化后模型

```shell
cd Paddle-Lite-Demo/libs
# 下载所需要的 Paddle Lite 预测库
sh download.sh
cd ../object_detection/assets
# 下载OPT 优化后模型
sh download.sh
cd ..
```

下载完成后会出现提示： `Extract done `

4、用 Xcode 打开 `picodet_demo/picodet_demo.xcodeproj` 文件，修改工程配置。依次修改 `General/Identity` 和 `Signing&Capabilities` 属性，替换为自己的工程代号和团队名称。（必须修改，不然无法通过编译），修改工程配置。

![Xcode1](https://paddlelite-demo.bj.bcebos.com/demo/image_classification/docs_img/ios/Xcode1.png)

![Xcode2](https://paddlelite-demo.bj.bcebos.com/demo/image_classification/docs_img/ios/Xcode2.png)

5、 IPhone 手机连接电脑，在 Xcode 中连接自己的手机 （第一次连接 IPhone 到电脑时，需要在 IPhone 的`设置->通用->设备管理`中选择本电脑并信任）

<p align="center"><img width="600" height="250"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode-phone.jpg"/>

5、按下左上角的 Run 按钮，自动编译 APP 并安装到手机。在苹果手机中设置信任该 APP（进入`设置->通用->设备管理`，选中新安装的 APP 并`验证该应用`）

  | APP 图标 | APP 效果 |
  | ---     | --- |
  | ![app_pic](https://paddlelite-demo.bj.bcebos.com/demo/object_detection/docs_img/ios/IOS_app.jpeg)    | ![app_res](https://paddlelite-demo.bj.bcebos.com/demo/object_detection/docs_img/ios/app_run_res.jpg) |

## IOS demo 结构讲解

1.  `third-party`： 存放预测库、模型、测试图片等相关信息
      * `assets`: 存放预测资源
        - models：模型文件，opt 工具转化后 Paddle Lite 模型
        - images：测试图片
        - labels：标签文件
      * `PaddleLite`：存放 Paddle Lite 预测库和头文件
        - lib
        - include
      * `opencv2.framework`：opencv  库和头文件

    ```shell
    # 位置：
    detection_demo/third-party/
    example：
    # IOS 预测库
    detection_demo/third-party/PaddleLite/lib/libpaddle_api_light_bundled.a
    # 预测库头文件
    detection_demo/third-party/PaddleLite/include/paddle_api.h
    detection_demo/third-party/PaddleLite/include/paddle_use_kernels.h
    detection_demo/third-party/PaddleLite/include/paddle_use_ops.h
    ```

 2.  `ViewController.mm`：主要预测代码

    ```shell
    # 位置
    detection_demo/ViewController.mm
    ``` 

  * `viewDidLoad`  方法
    APP 界面初始化、推理引擎 predictor 创建和运行方法，这个方法包含界面参数获取、predictor 构建和运行、图像前/后处理等内容
   
  * `processImage` 方法
    实现图像输入变化时，进行新的推理，并获取相应的输出结果

  * `pre_process` 方法
    输入预处理操作

  * `post_process` 方法
    输出后处理操作中


## 代码讲解 （如何使用 Paddle Lite `C++ API` 执行预测）

IOS 示例基于 `C++ API` 开发，调用 Paddle Lite `C++ API` 包括以下五步。更详细的 `API` 描述参考： [ Paddle Lite C++ API ](../api_reference/cxx_api_doc)。

```c++
#include <iostream>
// 引入 C++ API
#include "include/paddle_api.h"
#include "include/paddle_use_ops.h"
#include "include/paddle_use_kernels.h"

// 1. 设置 MobileConfig
MobileConfig config;
config.set_model_from_file(<modelPath>); // 设置 NaiveBuffer 格式模型路径
config.set_power_mode(LITE_POWER_NO_BIND); // 设置 CPU 运行模式
config.set_threads(4); // 设置工作线程数
// 如果需要使用 Metal 在 GPU 上加速预测，需要额外进行以下配置, 需将编译生成的 lite.metallib 拷贝到<metal_lib>路径下
config.set_metal_lib_path(<metal_lib>);
config.set_metal_use_mps(true);

// 2. 创建 PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

// 3. 设置输入数据
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
input_tensor->Resize({1, 3, 224, 224});
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}

// 4. 执行预测
predictor->run();

// 5. 获取输出数据
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
std::cout << "Output shape " << output_tensor->shape()[1] << std::endl;
for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
  std::cout << "Output[" << i << "]: " << output_tensor->data<float>()[i]
            << std::endl;
}
```

### Q&A:
问题：
- 提示 `feed kernel not found`:
  - 在包含 Paddle Lite 头文件的时候加上以下两行即可
  ```
   #include "include/paddle_use_ops.h"
   #include "include/paddle_use_kernels.h"
  ```
- 如果想用 FP16 模型推理：
  - 更新预测库：包含FP16 kernel的预测库，可以在 [release 官网](https://github.com/PaddlePaddle/Paddle-Lite/tags)下载，也可以参考[源码编译文档](../source_compile/macos_compile_ios)，自行编译。
  - 更新 nb 模型：需要使用 OPT 工具，将 `enable_fp16` 设置为 ON，重新转换模型。
  - FP16 预测库和 FP16 模型只在**V8.2 架构以上的手机**上运行
