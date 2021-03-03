# PaddleLite使用FPGA预测部署

Paddle Lite支持基于arm的FPGA zu3/zu5/zu9的模型预测，提供armv8的交叉编译

PaddleLite通过调用底层驱动实现对FPGA硬件的调度，目前只支持百度[Edgeboard开发板](https://ai.baidu.com/tech/hardware/deepkit)

![](https://paddlelite-data.bj.bcebos.com/doc_images/FPGA_demo/soft_arch.png)


## Lite实现FPGA简介

Lite支持FPGA作为后端硬件进行模型推理，其主要特性如下：

-  PaddleLite FPGA版本支持原生 fluid 模型，无须使用opt工具进行格式转化。
- Lite中FPGA的kernel（feed、fetch除外）均以FP16、NHWC的格式作为输入输出格式，所有的weights和bias仍为FP32、NCHW的格式，feed的输入和fetch的输出均为FP32、NCHW格式的数据，在提升计算速度的同时能做到用户对数据格式无感知
- 对于FPGA暂不支持的kernel，均会切回arm端运行，实现arm+FPGA混合布署运行
- 目前FPGA成本功耗都较低，Lite基于FPGA的模型性能远远好于arm端，可作为边缘设备首选硬件



## 已验证Paddle模型

分类网络：

* MobileNet 系列
   - MobileNetV1
   - MobileNetV2
* ResNet 系列
	- ResNet18
	- ResNet34
	- ResNet50
	- ResNet101
	- ResNet152
	- Res2Net50
	- SE-ResNet
* ResNext 系列
	- ResNext50
	- ResNext101
	- SE-ResNext
* Inception 系列
	- InceptionV3
	- InceptionV4
	

检测网络:

* SSD系列主干
	- Mobilenet-SSD
	- VGG-SSD
	- ResNet-SSD

* YOLO-V3 系列主干
	- Darknet50
	- MobileNet-V1
	- ResNet
	- tiny_yolo

分割网络:
MobileNet-deeplabV3 : coming soon


关键点网络：
HRNet : coming soon


## 准备工作

Edgeboard可以通过uart 串口线进行连接，也可以通过ssh进行连接，初次使用请参考[文档](https://ai.baidu.com/ai-doc/HWCE/Gkda62qno#edgeboard%E4%BC%A0%E8%BE%93%E6%96%87%E4%BB%B6%E6%96%B9%E5%BC%8F) 
Edgeboard 自带Samba服务器，可通过samba协议访问板上文件系统，进行数据拷贝。

## PaddleLite编译

需要提前准备带有FPGAdrv.ko的FPGA开发板（如edgeboard开发板）和Lite代码

CMAKE编译选项：

- 设置`LITE_WITH_FPGA=ON`和`LITE_WITH_ARM=ON`

其他编译选项与ARM编译相同，可以参考[“Paddle Lite在Docker下的ARM编译”](../source_compile/compile_linux)。
Lite提供FPGA编译脚本，位于lite/tools/build_FPGA.sh，在Lite根目录执行该脚本即可编译

示例如下：
```shell
    sh ./lite/tools/build_fpga.sh
    make publish_inference -j2
```

也可从Edgeboard[官网](https://ai.baidu.com/ai-doc/HWCE/Yk3b95s8o)下载最新的二进制更新库

## 应用编译
Edgeboard 自带 gcc, CMake, OpenCV 等工具和库，可直接在板子上进行编译，也可以在 Docker中进行交叉编译。

## 运行示例


我们提供了不同的示例工程
[示例工程下载链接](https://ai.baidu.com/ai-doc/HWCE/Yk3b95s8o)


以分类模型示例工程为例，工程目录结构如下

```bash
├── CMakeLists.txt // cmake 工程配置文件。
├── include //头文件
|   ├── commom.h   
├── configs // 配置文件目录
│   ├── Inceptionv2
│   │   └─ zebra.json //Inceptionv2配置文件（万分类-预置斑马识别）
│   ├── Inceptionv3
│   │   └─ zebra.json //Inceptionv3配置文件（千分类-预置斑马识别）
│   ├── mobilenetv1
│   │   └─ zebra.json //mobilenetv1配置文件（千分类-预置斑马识别）
│   └── resnet50
│       └─ drink.json //resnet50配置文件（三分类-预置矿泉水识别）
├── lib //(动态库放入系统内/usr/local/lib/paddle_lite/目录，此处为空文件夹)
├── models // 模型文件目录
│   ├── Inceptionv2
│   ├── Inceptionv3
│   ├── mobilenetv1
│   └── resnet50
│── src
│   ├── json.hpp // json 解析库
│   ├── video_detection.cpp // 视频推理示例
|   ├── image_detection.cpp // 图片推理示例
└── README.md
```
- **编译和执行示例工程**

```bash
# 连接开发板，并利用screen命令启动 [本机执行]
screen /dev/cu.SLAB_USBtoUART 115200
# 查看开发板ip并ssh登录到开发板，假设开发板ip为192.0.1.1 [本机执行]
ssh root@192.0.1.1

# 进入classification工程目录
cd /home/root/workspace/PaddleLiteSample/classification   
# 如果没有build目录，创建一个
mkdir build
# 打开build目录
cd build
# 调用cmake 创建 Makefile 
cmake ..
# 编译工程。
make

# 执行示例
./image_classify ../configs/resnet50/drink.json          

```


## 如何在Code中使用

在Lite中使用FPGA与ARM相似，具体的区别如下：

- 由于fpga运行模式为fp16精度、nhwc布局，所以需要修改相应的`valid_place`
- fpga不需要device的初始化和运行模式设置

代码示例：

```cpp
//构造places, FPGA使用以下几个places。
std::vector<Place> valid_places({
    Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},
    Place{TARGET(kHost), PRECISION(kFloat)},
    Place{TARGET(kARM), PRECISION(kFloat)},
});
//构造模型加载参数
paddle::lite_api::CxxConfig config;

if (combined_model) {
	//设置组合模型路径（两个文件）
    config.set_model_file(model_dir + "/model");
    config.set_param_file(model_dir + "/params");
} else {
	//设置模型目录路径，适用于一堆文件的模型
    config.set_model_dir(model_dir);
}

auto predictor = paddle::lite_api::CreatePaddlePredictor(config);

input->Resize({1, 3, height, width});
//获取tensor数据指针
auto* in_data = input->mutable_data<float>();
//图片读入相应数组当中
read_image(value, in_data);
//推理
predictor->Run();
//获取结果tensor，有多个结果时，可根据相应下标获取
auto output = predictor->GetOutput(0);
//获取结果数据
float *data = output->mutable_data<float>();
```
