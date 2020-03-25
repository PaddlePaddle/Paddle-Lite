
# Introduction
  我们都知道，PaddleLite可以做移动端预测，事实上PaddleLite支持在移动端做模型训练。本文给出使用PaddleLite做训练的例子，这一例子对应的任务是“波士顿房价预测”，又称作“fit-a-line”。
  
  你可以通过book库中的
[文档](https://paddlepaddle.org.cn/documentation/docs/zh/user_guides/simple_case/fit_a_line/README.cn.html)
和
[源码](https://github.com/PaddlePaddle/book/tree/develop/01.fit_a_line)
进一步了解“波士顿房价预测”这一任务的定义及其建模过程，
其使用线性回归（Linear Regression）
模型做建模。本文主要介绍如何将其迁移至Paddle-Lite进行训练。

# Requirements

- 一部安卓手机，用于运行训练程序
- 装了paddle的python

# Quick start

## Step1 build paddle-lite

请按照[paddle-lite官方文档](https://paddle-lite.readthedocs.io/zh/latest/user_guides/source_compile.html#paddlelite) 的教程编译full_publish的paddle-lite lib。具体的编译命令为：

```shell
./lite/tools/build.sh \
  --arm_os=android \
  --arm_abi=armv7 \
  --build_extra=ON \
  --arm_lang=gcc \
  --android_stl=c++_static \
  full_publish
```

## Step2 生成训练模型

```shell
python train.py --save_model
```

产物：

```shell
model_dir/
|-- fc_0.b_0
|-- fc_0.w_0
|-- learning_rate_0
`-- __model__

md5sum fc_0.w_0: 2c7b3649b2a9cf7bcd19f8b256ce795d
md5sum __model__: d4c5a1458cd479dd36c51a5c6c383a3c
Paddle version: 137987534ebf06f47974b76165c3050ee687fccd
```

## Step3 准备训练数据

```shell
wget http://paddlemodels.bj.bcebos.com/uci_housing/housing.data
```

## Step4 编译lr_trainer

```shell
cd cplus_train/
sh run_build.sh /path/to/your/Paddle-Lite/build.lite.android.armv7.gcc/ /path/to/your/android-ndk-r17c
```

产物:
```shell
bin/
`-- demo_trainer
```

## Step4 执行训练

请用USB连接线连接手机和笔记本，在笔记本的终端执行以下命令，将模型和数据推至手机上运行：

```
local_path=/data/local/tmp
adb shell "mkdir "${local_path}
adb push libpaddle_full_api_shared.so ${local_path}
adb push housing.data ${local_path}
adb push model_dir ${local_path}
adb push demo_trainer ${local_path}
adb shell chmod +x ${local_path}/demo_trainer
adb shell "export LD_LIBRARY_PATH="${local_path}" && export LIBRARY_PATH="${local_path}" && cd "${local_path}" && ./demo_trainer true"
```

期望结果：

```
sample 0: Loss: 564.317
sample 1: Loss: 463.9
sample 2: Loss: 1197.54
sample 3: Loss: 1093.83
sample 4: Loss: 1282.76
sample 5: Loss: 792.097
sample 6: Loss: 491.776
sample 7: Loss: 698.496
sample 8: Loss: 248.445
sample 9: Loss: 325.135
```

# 与Paddle训练结果做校对

## 前10个Loss值

为了验证paddle与lite的一致性，我们控制模型参数一致、数据一致、batch size = 1的情况下，训练10个batch， 记录了二者的loss值。

python + paddle 命令:

```shell
  fluid train.py --num_steps=10 --batch_size=1
```

python + paddle 结果:

```shell
Train cost, Step 0, Cost 564.317017
Train cost, Step 1, Cost 463.900238
Train cost, Step 2, Cost 1197.537354
Train cost, Step 3, Cost 1093.833008
Train cost, Step 4, Cost 1282.760254
Train cost, Step 5, Cost 792.097351
Train cost, Step 6, Cost 491.775848
Train cost, Step 7, Cost 698.496033
Train cost, Step 8, Cost 248.444885
Train cost, Step 9, Cost 325.135132
```

c++ 与 paddle-lite命令：
```
./demo_trainer true
```

c++ 与 paddle-lite结果：
```
sample 0: Loss: 564.317
sample 1: Loss: 463.9
sample 2: Loss: 1197.54
sample 3: Loss: 1093.83
sample 4: Loss: 1282.76
sample 5: Loss: 792.097
sample 6: Loss: 491.776
sample 7: Loss: 698.496
sample 8: Loss: 248.445
sample 9: Loss: 325.135
```

## Loss 曲线

控制训练时的batch size为20，每个epoch对训练数据做全局shuffle，训练100个epoch后，paddle和lite的loss曲线对比如下。

![lr_loss](image/lr_loss.png)

如果想复现上述效果，paddle+python的运行命令为：

```
git clone https://github.com/PaddlePaddle/book.git
cd book/01.fit_a_line
python train.py
```

lite + c++的运行命令为：
```
./demo_trainer false
```
