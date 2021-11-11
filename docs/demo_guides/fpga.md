# 百度 EdgeBoard 部署示例

Paddle Lite 支持基于 ARM 的 FPGA zu3/zu5/zu9 的模型预测，提供 armv8 的交叉编译

Paddle Lite 通过调用底层驱动实现对 FPGA 硬件的调度，目前只支持百度[ Edgeboard 开发板](https://ai.baidu.com/tech/hardware/deepkit)

![](https://paddlelite-data.bj.bcebos.com/doc_images/FPGA_demo/soft_arch.png)


## Paddle Lite 实现 FPGA 简介

Paddle Lite 支持 FPGA 作为后端硬件进行模型推理，其主要特性如下：

-  Paddle Lite FPGA 版本支持原生 fluid 模型，无须使用 opt 工具进行格式转化。
- Paddle Lite 中 FPGA 的 kernel（feed、fetch 除外）均以 `FP16`、`NHWC` 的格式作为输入输出格式，所有的 `weights` 和 `bias` 仍为 `FP32`、`NCHW` 的格式，feed 的输入和 fetch 的输出均为 `FP32`、`NCHW` 格式的数据，在提升计算速度的同时能做到用户对数据格式无感知
- 对于 FPGA 暂不支持的 kernel，均会切回 ARM 端运行，实现 ARM+FPGA 混合布署运行
- 目前 FPGA 成本功耗都较低，Paddle Lite 基于 FPGA 的模型性能远远好于 ARM 端，可作为边缘设备首选硬件



## 已验证 Paddle 模型

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

* SSD 系列主干
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

Edgeboard 可以通过 uart 串口线进行连接，也可以通过 `ssh` 进行连接，初次使用请参考[文档](https://ai.baidu.com/ai-doc/HWCE/Gkda62qno#edgeboard%E4%BC%A0%E8%BE%93%E6%96%87%E4%BB%B6%E6%96%B9%E5%BC%8F) 
Edgeboard 自带 Samba 服务器，可通过 samba 协议访问板上文件系统，进行数据拷贝。

## Paddle Lite 编译

需要提前准备带有 FPGAdrv.ko 的 FPGA 开发板（如 Edgeboard 开发板）和 Paddle Lite 代码

CMAKE 编译选项：

- 设置 `LITE_WITH_FPGA=ON` 和 `LITE_WITH_ARM=ON`

其他编译选项与 ARM 编译相同，可以参考[Paddle Lite 在 Docker 下的 ARM 编译](../source_compile/docker_env)。
Paddle Lite 提供 FPGA 编译脚本，位于 `lite/tools/build_FPGA.sh`，在 Paddle Lite 根目录执行该脚本即可编译

示例如下：
```shell
    sh ./lite/tools/build_fpga.sh
    make publish_inference -j2
```

也可从 Edgeboard [官网](https://ai.baidu.com/ai-doc/HWCE/Yk3b95s8o)下载最新的二进制更新库

## 应用编译
Edgeboard 自带 gcc, CMake, OpenCV 等工具和库，可直接在板子上进行编译，也可以在 Docker 中进行交叉编译。

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
│   │   └─ zebra.json // Inceptionv2 配置文件（万分类-预置斑马识别）
│   ├── Inceptionv3
│   │   └─ zebra.json // Inceptionv3 配置文件（千分类-预置斑马识别）
│   ├── mobilenetv1
│   │   └─ zebra.json // mobilenetv1 配置文件（千分类-预置斑马识别）
│   └── resnet50
│       └─ drink.json // resnet50 配置文件（三分类-预置矿泉水识别）
├── lib //(动态库放入系统内 /usr/local/lib/paddle_lite/ 目录，此处为空文件夹)
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
# 连接开发板，并利用 screen 命令启动 [本机执行]
screen /dev/cu.SLAB_USBtoUART 115200
# 查看开发板 ip 并 ssh 登录到开发板，假设开发板 ip 为 192.0.1.1 [本机执行]
ssh root@192.0.1.1

# 进入 classification 工程目录
cd /home/root/workspace/PaddleLiteSample/classification   
# 如果没有 build 目录，创建一个
mkdir build
# 打开 build 目录
cd build
# 调用 cmake 创建 Makefile 
cmake ..
# 编译工程。
make

# 执行示例
./image_classify ../configs/resnet50/drink.json          

```


## 如何在 Code 中使用

在Paddle Lite 中使用 FPGA 与 ARM 相似，具体的区别如下：

- 由于 FPGA 运行模式为 `FP16` 精度、`NHWC` 布局，所以需要修改相应的 `valid_place`
- FPGA 不需要 device 的初始化和运行模式设置

代码示例：

```cpp
// 构造 places, FPGA 使用以下几个 places。
std::vector<Place> valid_places({
    Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},
    Place{TARGET(kHost), PRECISION(kFloat)},
    Place{TARGET(kARM), PRECISION(kFloat)},
});
// 构造模型加载参数
paddle::lite_api::CxxConfig config;

if (combined_model) {
	// 设置组合模型路径（两个文件）
    config.set_model_file(model_dir + "/model");
    config.set_param_file(model_dir + "/params");
} else {
	// 设置模型目录路径，适用于一堆文件的模型
    config.set_model_dir(model_dir);
}

auto predictor = paddle::lite_api::CreatePaddlePredictor(config);

input->Resize({1, 3, height, width});
// 获取 tensor 数据指针
auto* in_data = input->mutable_data<float>();
// 图片读入相应数组当中
read_image(value, in_data);
// 推理
predictor->Run();
// 获取结果tensor，有多个结果时，可根据相应下标获取
auto output = predictor->GetOutput(0);
// 获取结果数据
float *data = output->mutable_data<float>();
```
