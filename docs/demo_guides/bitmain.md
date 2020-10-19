# PaddleLite使用Bitmain：Sophon BM1682/BM1684 预测部署

Paddle Lite已支持在比特大陆的Sophon BM1682/BM1684处理器上进行预测部署。


## 支持现状

### 已支持的芯片

- Sophon BM1682
- Sophon BM1684

### 已支持的Paddle模型

- [Mobilenet](https://paddlelite-demo.bj.bcebos.com/mobilenet_v1.tar.gz)
- [Yolov3]
- [Mobilenet-ssd]
- [Inceptionv4](https://paddlelite-demo.bj.bcebos.com/inception_v4_simple.tar.gz)
- [Vgg16]
- [DarkNet-YOLOv3]
- [PyramidBox]
- 百度内部业务模型（由于涉密，不方便透露具体细节）

### 已支持（或部分支持）的Paddle算子

- relu
- leaky_relu
- sqrt
- square
- sigmoid
- assign_value
- batch_norm
- box_coder
- cast
- concat
- conv2d
- depthwise_conv2d
- conv2d_transpose
- depthwise_conv2d_transpose
- density_prior_box
- dropout
- elementwise_add
- elementwise_mul
- elementwise_sub
- elementwise_div
- fill_constant
- nearest_interp
- bilinear_interp
- matmul
- mul
- multiclass_nms
- norm
- pool2d
- max_pool2d_with_index
- prior_box
- reduce_sum
- reduce_mean
- reduce_max
- reshape
- flatten
- scale
- shape
- slice
- softmax
- split
- swish
- transpose
- yolo_box

## 参考示例演示

### 测试设备(Sophon BM1682)

### 准备设备环境

- 将Sophon BM1682或者BM1684 处理器安装到主机上后，下载对应的硬件驱动

### 准备本地编译环境

1. Docker 容器环境；
2. Linux（推荐 Ubuntu 16.04）环境。

请先根据[编译环境准备](compile_env)中的内容，根据您的开发环境安装编译预测库所需的编译环境
 **NOTE：** 在创建docker容器时，设置好挂载device的路径。例如：sudo docker run -it --name work_bm -v $PWD:/code --device=/dev/bm1682-dev0:/dev/bm1682-dev0 --device=/dev/bmdev-ctl:/dev/bmdev-ctl --net=host 1423ff1080e5 /bin/bash）

### 编译Paddle-Lite工程

1. 下载代码
  ```
  git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  ```

2. 编译
  ```
  # 进入代码目录
  cd Paddle-Lite

  # 运行编译脚本
  ./lite/tools/build_bm.sh --target_name=BM1682 --test=ON

  # 编译结束会在本目录下生成 build.lite.bm 目录


  # 测试的demo在build.lite.bm/inference_lite_lib/demo/cxx/bm_demo/目录下
  ```
3. 编译结果说明
  ```
  # 编译生成的库目录
  build.lite.bm/lite/api

  # 运行编译脚本
  ./lite/tools/build_bm.sh --target_name=BM1682 --test=ON

  # 编译结束会在本目录下生成 build.lite.bm 目录

  ```

### 运行demo

```
  # 测试的demo在build.lite.bm/inference_lite_lib/demo/cxx/bm_demo/目录下
  打开Demo文件夹，运行build.sh
  ./build.sh

  # build.lite.bm/inference_lite_lib/demo/cxx/bm_demo/目录下的lib和include文件夹
  lib：部署demo需要的动态库
  include：需要的头文件
```


## 其它说明

- 如需更进一步的了解相关产品的信息，请联系欧阳剑ouyangjian@baidu.com；
- 百度昆仑的研发同学正在持续适配更多的Paddle算子，以便支持更多的Paddle模型。
