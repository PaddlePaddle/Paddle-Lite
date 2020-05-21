# PaddleLite使用CUDA预测部署

Lite支持在x86_64，arm64架构上（如：TX2）进行CUDA的编译运行。

## 编译

**NOTE：** 如果是在TX2等NVIDIA嵌入式硬件上编译，请使用最新的[Jetpack](https://developer.nvidia.com/embedded/jetpack) 安装依赖库。


一： 下载代码

```
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
```

二：编译

```
# 进入代码目录
cd Paddle-Lite

# 运行编译脚本
# 编译结束会在本目录下生成 build_cuda 目录
# 编译过程中如果提示找不到CUDA，CUDNN，请在环境变量设置CUDA_TOOLKIT_ROOT_DIR, CUDNN_ROOT
# CUDA_TOOLKIT_ROOT_DIR，CUDNN_ROOT分别表示CUDA，CUDNN的根目录
./lite/tools/build.sh cuda
# 如果使用python接口，需要打开build_python选项
./lite/tools/build.sh --build_python=ON cuda
```

## 编译结果说明

cuda的编译结果位于 `build_cuda/inference_lite_lib`
**具体内容**说明：

1、 `bin`文件夹：可执行工具文件，目前为空

2、 `cxx`文件夹：包含c++的库文件与相应的头文件

- `include`  : 头文件
- `lib` : 库文件
  - 打包的静态库文件：
    - `libpaddle_api_full_bundled.a`  ：包含 full_api 和 light_api 功能的静态库
  - 打包的动态态库文件：
    - `libpaddle_full_api_shared.so` ：包含 full_api 和 light_api 功能的动态库

3、 `third_party` 文件夹：第三方库文件

4、 `demo` 文件夹：c++ demo.

如果编译打开了python选项，则会在 `build_cuda/inference_lite_lib/python/lib/` 目录下生成 `lite.so`。

## 运行

以下以Yolov3模型为例，介绍如何在Nvidia GPU硬件上运行模型。

一： 下载darknet_yolov3模型，模型信息请参考[这里](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/yolov3)

```
# 下载模型
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/yolov3_infer.tar.gz
tar -zxf yolov3_infer.tar.gz
# 下载图片样例
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/kite.jpg
```

二： 运行   

**NOTE：** 此处示例使用的是python接口。

``` python
#-*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
import cv2
sys.path.append('build_cuda/inference_lite_lib/python/lib')
from lite import *

def read_img(im_path, resize_h, resize_w):
  im = cv2.imread(im_path).astype('float32')
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  h, w, _ = im.shape
  im_scale_x = resize_h / float(w)
  im_scale_y = resize_w / float(h)
  out_img = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_CUBIC)
  mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, -1))
  std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, -1))
  out_img = (out_img / 255.0 - mean) / std
  out_img = out_img.transpose((2, 0, 1))
  return out_img

# 配置config
a = CxxConfig()
a.set_model_file('./yolov3_infer/__model__') # 指定模型文件路径 
a.set_param_file('./yolov3_infer/__params__') # 指定参数文件路径
place_cuda = Place(TargetType.CUDA)
a.set_valid_places([place_cuda])

# 创建predictor
predictor = create_paddle_predictor(a)

# 设置输入
input_tensor = predictor.get_input(0);
height, width = 608, 608
input_tensor.resize([1, 3, height, width])
data = read_img('./kite.jpg', height, width).flatten()
input_tensor.set_float_data(data, TargetType.CUDA)

in2 = predictor.get_input(1);
in2.resize([1, 2])
in2.set_int32_data([height, width], TargetType.CUDA)

# 运行
predictor.run()

# 获取输出
output_tensor = predictor.get_output(0);

print (output_tensor.shape())
# [100L, 6L]
print (output_tensor.target())
# TargetType.Host
print (output_tensor.float_data()[:6])
# [0.0, 0.9862784743309021, 98.51927185058594, 471.2381286621094, 120.73092651367188, 578.33251953125]

```

**NOTE：** 此处示例使用的是C++接口。

```
cd build_cuda/inference_lite_lib/demo/cxx/
mkdir build && cd build
cmake ..
make
wget https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/yolov3_infer.tar.gz
tar -zxf yolov3_infer.tar.gz
./demo yolov3_infer
```
