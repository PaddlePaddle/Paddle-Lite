# PaddleLite使用FPGA预测部署

Paddle Lite支持基于arm的FPGA zu3/zu5/zu9的模型预测，提供armv8的交叉编译

Lite基于FPGA运行模型需要相应的FPGA驱动，目前只支持百度[Edgeboard开发板](https://ai.baidu.com/tech/hardware/deepkit)

## Lite实现FPGA简介

Lite支持FPGA作为后端硬件进行模型推理，其主要特性如下：

- Lite中FPGA的kernel（feed、fetch除外）均以FP16、NHWC的格式作为输入输出格式，所有的weights和bias仍为FP32、NCHW的格式，feed的输入和fetch的输出均为FP32、NCHW格式的数据，在提升计算速度的同时能做到用户对数据格式无感知

- 对于FPGA暂不支持的kernel，均会切回arm端运行，实现arm+FPGA混合布署运行

- 目前FPGA成本功耗都较低，Lite基于FPGA的模型性能远远好于arm端，可作为边缘设备首选硬件

## 编译

需要提前准备带有FPGAdrv.ko的FPGA开发板（如edgeboard开发板）和Lite代码

CMAKE编译选项：

- 设置`LITE_WITH_FPGA=ON`和`LITE_WITH_ARM=ON`

其他编译选项与ARM编译相同，可以参考[“Paddle Lite在Docker下的ARM编译”](../user_guides/source_compile)。
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
        -DLITE_WITH_FPGA=ON \
        -DARM_TARGET_OS=armlinux 
    make publish_inference -j2
```
Lite提供FPGA编译脚本，位于lite/tools/build_FPGA.sh，在Lite根目录执行该脚本即可编译

## 运行示例

- **运行文件准备**

下面以Resnet50模型为例，介绍如何使用edgeboard开发板实现模型运行

```bash
#连接开发板，并利用screen命令启动 [本机执行]
screen /dev/cu.SLAB_USBtoUART 115200
#查看开发板ip并ssh登录到开发板，假设开发板ip为192.0.1.1 [本机执行]
ssh root@192.0.1.1

#在开发板上建立目录workspace，拷贝FPGA驱动FPGAdrv.ko到workspace目录 [开发板执行]
mkdir workspace && scp $DRIVER_PATH/FPGAdrv.ko workspace

#将Lite中编译好的测试程序拷贝到开发板workspace目录 [本机执行]
scp $LITE_ROOT/build_FPGA/lite/api/test_resnet50_FPGA root@$EDGEBOARD_IP:workspace/
#把Resnet50的模型和参数scp到开发板workspace目录 [本机执行]
scp -r $LITE_ROOT/build_FPGA/lite/third_party/install/resnet50/ root@$EDGEBOARD_IP:workspace/

#在运行模型前需要加载FPGA驱动 [开发板执行]
insmod FPGAdrv.ko
#给测试程序添加可运行权限 [开发板执行]
chmod +x test_resnet50_FPGA
```

- **使用FPGA进行模型预测**

```bash
#以下命令均在开发板上运行
#直接运行单测程序
./test_resnet50_FPGA --model_dir=resnet50
#如果需要测试性能，可以用repeats参数设置模型运行次数（如1000），同时可以设置预热次数（如10）来让硬件事先运行到稳定水平
./test_resnet50_FPGA --model_dir=resnet50 --repeats=1000 --warmup=10
```

## 如何在Code中使用

在Lite中使用FPGA与ARM相似，具体的区别如下：

- 由于fpga运行模式为fp16精度、nhwc布局，所以需要修改相应的`valid_place`
- fpga不需要device的初始化和运行模式设置

代码示例：
```cpp
lite::Predictor predictor;
std::vector<Place> valid_places(
      {Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},Place{TARGET(kARM)});

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
