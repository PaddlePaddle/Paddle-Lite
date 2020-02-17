
# 预测库说明

Paddle-Lite的编译结果为预测库文件（包括静态库和动态库），具体编译过程参考[源码编译](./source_compile)。

Lite预测库分为**基础预测库**和**全量预测库**：基础预测库只打包了基础模型需要的基础算子，预测库体积较小；全量预测库打包了所有的Lite算子，可以支持更多的模型，但是预测库的体积也更大。 编译时由编译选项 `build_extra`(默认为OFF)控制，`--build_extra=OFF`时编译基础预测库，`--build_extra=ON`时编译全量的预测库。

## 基础预测库

### 编译方法
编译时设置`--build_extra=OFF` (默认值) 或不指定即可编译出基础预测库。例如：

```
./lite/tools/build.sh  --arm_os=android  --arm_abi=armv8 --arm_lang=gcc  --android_stl=c++_static  tiny_publish
```

### 基础预测库支持的功能

（1）支持基础CV模型

（2）支持基础的in8量化模型

（3）支持[benchmark测试](../benchmark/benchmark)


### 基础预测库支持的基础模型：

1. fluid基础模型（paddle model 提供的基础模型9个）

```
mobileNetV1     mnasnet     yolov3   ssd_mobilenetv1    shufflenet_v2
mobileNetV2     resnet50    unet     squeezenet_v11
```

2. int8量化模型模型

```
mobilenet_v1   mobilenet_v2   resnet50
```

### 特点
  轻量级预测库，体积更小，支持常用的基础模型。



## 全量预测库

### 编译方法
编译时设置`--build_extra=ON` 即可编译出全量预测库。例如：

```
./lite/tools/build.sh  --arm_os=android  --arm_abi=armv8 --arm_lang=gcc  --android_stl=c++_static --build_extra=ON tiny_publish
```
### 全量预测库功能

（1） 基础预测库所有功能

（2）支持所有Paddle-Lite中注册的所有算子

### 特点
  支持更多的硬件平台和算子，可以支持更多模型但体量更大。