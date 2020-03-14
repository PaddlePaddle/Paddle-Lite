
# `build_extra`参数说明：

Lite预测库分为**基础预测库**和**全量预测库(with_extra)**：基础预测库只包含基础CV算子（OP），体积较小；全量预测库包含所有Lite算子，体积较大，支持模型较多。

编译时由编译选项 `build_extra`(默认为OFF)控制，`--build_extra=OFF`时编译**基础预测库**，`--build_extra=ON`时编译**全量预测库**。

## 基础预测库( [基础OP列表](../advanced_user_guides/support_operation_list.html#basic-operators) )


### 支持功能

（1）87个[基础OP](../advanced_user_guides/support_operation_list.html#basic-operators)       （2）9个基础模型       （3）3个in8量化模型


### 支持的模型

1. fluid基础模型（来源：[paddle-models](https://github.com/PaddlePaddle/models) ）

```
mobilenetV1     mnasnet     yolov3   ssd_mobilenetv1    shufflenet_v2
mobilenetV2     resnet50    unet     squeezenet_v11
```

2. int8量化模型

```
mobilenet_v1   mobilenet_v2   resnet50
```

### 特点
  轻量级预测库，体积更小，支持常用模型。

### 编译方法
编译时设置`--build_extra=OFF` (默认值) 编译出基础预测库。例如：

```
./lite/tools/build.sh  --arm_os=android  --arm_abi=armv8 --arm_lang=gcc  --android_stl=c++_static  tiny_publish
```


## 全量预测库( [OP列表](../advanced_user_guides/support_operation_list.html#op) )


### 支持功能

   Paddle-Lite中的全量算子（ [基础OP](../advanced_user_guides/support_operation_list.html#basic-operators) + [Extra OP](../advanced_user_guides/support_operation_list.html#extra-operators-build-extra-on) ）

### 特点
   包含更多算子、支持更多模型，但体量更大。

### 编译方法
设置`--build_extra=ON` 可编译出全量预测库。例如：

```
./lite/tools/build.sh  --arm_os=android  --arm_abi=armv8 --arm_lang=gcc  --android_stl=c++_static --build_extra=ON tiny_publish
```
