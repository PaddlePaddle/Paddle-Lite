# PaddleLite使用IntelFPGA预测部署

Paddle Lite支持基于arm的IntelFPGA C5的模型预测，提供armv7hf的交叉编译

PaddleLite通过调用底层驱动实现对FPGA硬件的调度，以及对应的API接口。

## Lite实现IntelFPGA简介

Lite支持IntelFPGA作为后端硬件进行模型推理，其主要特性如下：

- Lite中IntelFPGA的kernel均以FP32、NCHW的格式作为输入输出格式

- 对于IntelFPGA暂不支持的kernel，均会切回ARM端运行，实现ARM+FPGA混合布署运行

## 支持芯片
- [Cyclone V](https://www.intel.cn/content/dam/altera-www/global/en_US/pdfs/literature/hb/cyclone-v/cv_51002.pdf)

### 已支持（或部分支持）的Paddle算子

- relu/relu6/leakyrelu
- conv2d
- depthwise_conv2d

### 已支持的Paddle模型

- [SSD_MobileNet_V1](https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_coco_pretrained.tar)

## 编译

需要提前准备带有intelfpgadrv.ko的IntelFPGA开发板C5MB/C5TB和Lite代码

CMAKE编译选项：

- 设置`LITE_WITH_INTEL_FPGA=ON`和`LITE_WITH_ARM=ON`

其他编译选项与ARM编译相同，可以参考[“Paddle Lite在Docker下的ARM编译”](../source_compile/compile_linux)。

示例如下：
```shell
    cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DLITE_WITH_OPENMP=ON   \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=OFF \
        -DLITE_WITH_INTEL_FPGA=ON \
        -DARM_TARGET_OS=armlinux 
    make publish_inference -j2
```
Lite提供IntelFPGA编译脚本，位于lite/tools/build_intel_fpga.sh full_publish，在Lite根目录执行该脚本即可编译

## 运行示例

- **运行文件准备**

下面以SSD模型为例，介绍如何使用C5MB/C5TB开发板实现模型运行

```bash
#打开串口调试工具，如Putty或SecureCRT，选择对应的调试串口，并设置串口属性，
#波特率：115200，数据位：8，停止位：1，奇偶校验：无[主机上执行]
#上电C5MB开发板，并在串口调试工具中登录
awcloud login: root
Password: #密码：Awcloud@123
#进入/opt目录[开发板执行]
cd /opt
#在运行模型前需要加载FPGA驱动[开发板执行]
insmod driver/intelfpgadrv.ko
```

- **使用IntelFPGA进行模型预测**

```bash
#以下命令均在开发板上运行，在开发板上已经部署了对应的输入图片，模型，驱动程序，执行程序等
#运行SSD测试程序，输入图片为/opt/images/dog.jpg，输出图片为/opt/dog_result.jpg
./run_ssd.sh
```

## 如何在Code中使用

在Lite中使用IntelFPGA与ARM相似，具体的区别如下：

- 由于IntelFPGA运行模式为FP32精度、NCHW布局，所以需要修改相应的`valid_place`

代码示例：
```cpp
lite::Predictor predictor;
std::vector<Place> valid_places(
      {Place{TARGET(kIntelFPGA), PRECISION(kFloat), DATALAYOUT(kNCHW)},Place{TARGET(kARM)});

predictor.Build(model_dir, "", "", valid_places);

auto* input_tensor = predictor.GetInput(0);
input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
auto* data = input_tensor->mutable_data<float>();
auto item_size = input_tensor->dims().production();
//假设设置输入数据全为1
for (int i = 0; i < item_size; i++) {
  data[i] = 1;
}

predictor.Run();
auto* out = predictor.GetOutput(0);
```
