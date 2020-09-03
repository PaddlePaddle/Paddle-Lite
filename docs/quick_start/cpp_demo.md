# C++ 完整示例

本章节包含2部分内容：(1) [C++ 示例程序](cpp_demo.html#id1); (2) [C++ 应用开发说明](cpp_demo.html#id11)。

## C++ 示例程序

本章节展示的所有C++ 示例代码位于 [demo/c++](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/cxx) 。

### 1. 环境准备

要编译和运行Android C++ 示例程序，你需要准备：

1. 一台armv7或armv8架构的安卓手机
2. 一台可以编译PaddleLite的电脑

### 2. 下载预编译的预测库

预测库下载界面位于[Lite预编译库下载](release_lib)，可根据您的手机型号选择合适版本。

以**Android-ARMv8架构**为例，可以下载以下版本：

| Arch  |with_extra|arm_stl|with_cv|下载|
|:-------:|:-----:|:-----:|:-----:|:-------:|
|armv8|OFF|c++_static|OFF|[release/v2.6.1](https://paddlelite-data.bj.bcebos.com/Release/2.6.1/Android/inference_lite_lib.android.armv8.gcc.c++_static.CV_OFF.tar.gz)|

**解压后内容结构如下：**

```shell
inference_lite_lib.android.armv8          Paddle-Lite 预测库
├── cxx                                       C++ 预测库
│   ├── include                                   C++ 预测库头文件
│   └── lib                                       C++ 预测库文件
│       ├── libpaddle_api_light_bundled.a             静态预测库
│       └── libpaddle_light_api_shared.so             动态预测库
├── demo                                      示例 Demo
│   ├── cxx                                       C++ 示例 Demo
│   └── java                                      Java 示例 Demo
└── java                                      Java 预测库
```

### 3. 准备预测部署模型

(1) 模型下载：下载[mobilenet_v1](http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz)模型后解压，得到Paddle非combined形式的模型，位于文件夹 `mobilenet_v1` 下。可通过模型可视化工具[Netron](https://lutzroeder.github.io/netron/)打开文件夹下的`__model__`文件，查看模型结构。

```shell
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxf mobilenet_v1.tar.gz
```

(2) 模型转换：Paddle的原生模型需要经过[opt](../user_guides/model_optimize_tool)工具转化为Paddle-Lite可以支持的naive_buffer格式。

方式一: 下载[opt工具](../user_guides/model_optimize_tool)，放入与`mobilenet_v1`文件夹同级目录，终端输入以下命令转化模型

```shell
# Linux
wget https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.6.1/opt
chmod +x opt
./opt --model_dir=./mobilenet_v1 \
      --optimize_out_type=naive_buffer \
      --optimize_out=./mobilenet_v1_opt

# Mac
wget https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.6.1/opt_mac
chmod +x opt_mac
./opt_mac --model_dir=./mobilenet_v1 \
          --optimize_out_type=naive_buffer \
          --optimize_out=./mobilenet_v1_opt
```

方式二: 通过pip安装paddlelite，终端输入命令转化模型

```shell
python -m pip install paddlelite
paddle_lite_opt --model_dir=./mobilenet_v1 \
                --optimize_out_type=naive_buffer \
                --optimize_out=./mobilenet_v1_opt
```

以上命令执行成功之后将在同级目录生成名为`mobilenet_v1_opt.nb`的优化后模型文件。



### 4. 编译预测示例程序

准备好预测库和模型，就可以直接编译随着预测库一起发布的 C++ Demo，位于在第二步中下载的预测库文件目录下`inference_lite_lib.android.armv8/demo/cxx`。以mobilenet_v1为例，目录下的`mobile_light`为mobilenet_v1预测示例，预测程序需要编译为Android可执行文件。

```shell
cd inference_lite_lib.android.armv8/demo/mobile_light
make
```

会在同级目录下生成名为`mobilenetv1_light_api`的可执行文件。

### 5. 预测部署和执行

(1) 设置手机：手机USB连接电脑，打开`设置 -> 开发者模式 -> USB调试 -> 允许（授权）当前电脑调试手机`。保证当前电脑已经安装[adb工具](https://developer.android.com/studio/command-line/adb)，运行以下命令，确认当前手机设备已被识别：

``` shell
adb devices
# 如果手机设备已经被正确识别，将输出如下信息
List of devices attached
017QXM19C1000664	device
```

(2) 预测部署：第二步中的C++动态预测库文件`libpaddle_light_api_shared.so`，将第三步中生成的优化后模型文件`mobilenet_v1_opt.nb`和第四步中编译得到的预测示例程序`mobilenetv1_light_api`放入同一文件夹，并将这三个文件推送到手机：

``` shell
chmod +x mobilenetv1_light_api
adb push mobilenet_v1_opt.nb /data/local/tmp
adb push libpaddle_light_api_shared.so /data/local/tmp
adb push mobilenetv1_light_api /data/local/tmp

# 如果推送成功，将显示如下信息
adb shell 'ls -l /data/local/tmp'
total 24168
-rwxrwxrwx 1 root root  1624280 2020-09-01 13:47 libpaddle_light_api_shared.so
-rw-rw-rw- 1 root root 17018243 2020-09-01 12:28 mobilenet_v1_opt.nb
-rwxrwxrwx 1 root root  6076144 2020-09-01 13:47 mobilenetv1_light_api
```

(3) 执行预测，以下输出为mobilenet_v1模型在全1输入时，得到的预测结果。

```shell
adb shell 'cd /data/local/tmp && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp && ./mobilenetv1_light_api mobilenet_v1_opt.nb'

# 如果正确运行，将输出如下信息
run_idx:1 / 10: 33.821 ms
run_idx:2 / 10: 33.8 ms
run_idx:3 / 10: 33.867 ms
run_idx:4 / 10: 34.009 ms
run_idx:5 / 10: 33.699 ms
run_idx:6 / 10: 33.644 ms
run_idx:7 / 10: 33.611 ms
run_idx:8 / 10: 33.783 ms
run_idx:9 / 10: 33.731 ms
run_idx:10 / 10: 33.423 ms

======= benchmark summary =======
input_shape(NCHW):1 3 224 224
model_dir:mobilenet_v1_opt.nb
warmup:10
repeats:10
max_duration:34.009
min_duration:33.423
avg_duration:33.7388

====== output summary ======
output tensor num:1

--- output tensor 0 ---
output shape(NCHW):1 1000
output tensor 0 elem num:1000
output tensor 0 standard deviation:0.00219646
output tensor 0 mean value:0.001
```

### 更多C++示例

#### 图像分类示例

```shell
cd inference_lite_lib.android.armv8/demo/cxx/mobile_classify

# 下载模型
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxvf mobilenet_v1.tar.gz
# 转化模型
paddle_lite_opt --model_dir=./mobilenet_v1 \
                --optimize_out_type=naive_buffer \
                --optimize_out=./mobilenet_v1_opt
# 编译预测程序
make
# 预测部署
adb push mobile_classify /data/local/tmp/
adb push mobilenet_v1_opt.nb /data/local/tmp/
adb push mobilenet_v1/test.jpg /data/local/tmp/
adb push mobilenet_v1/labels.txt /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell 'chmod +x /data/local/tmp/mobile_classify'
# 执行预测
adb shell 'cd /data/local/tmp && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp &&  ./mobile_classify mobilenet_v1_opt.nb test.jpg labels.txt'

# 运行成功后 ，将在控制台输出预测结果的前5个类别的类型索引、名字和预测概率
parameter:  model_file, image_path and label_file are necessary
parameter:  topk, input_width,  input_height, are optional
i: 0, index: 287, name:  lynx, catamount, score: 0.317595
i: 1, index: 285, name:  Egyptian cat, score: 0.308135
i: 2, index: 281, name:  tabby, tabby cat, score: 0.161924
i: 3, index: 282, name:  tiger cat, score: 0.093659
i: 4, index: 283, name:  Persian cat, score: 0.060198
```

#### 目标检测示例

```shell
cd inference_lite_lib.android.armv8/demo/cxx/ssd_detection

# 下载模型
wget https://paddlelite-data.bj.bcebos.com/doc_models/ssd_mobilenet_v1.tar.gz
tar zxvf ssd_mobilenet_v1.tar.gz
# 转化模型
paddle_lite_opt --model_dir=./ssd_mobilenet_v1 \
                --optimize_out_type=naive_buffer \
                --optimize_out=./ssd_mobilenet_v1_opt
# 编译预测程序
make
# 预测部署
adb push ssd_detection /data/local/tmp/
adb push ssd_mobilenet_v1_opt.nb /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell 'chmod +x /data/local/tmp/ssd_detection'
# 执行预测
adb shell 'cd /data/local/tmp && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp && ./ssd_detection ssd_mobilenet_v1_opt.nb test.jpg'

# 运行成功后 ，将在控制台输出检测目标的类型、预测概率和坐标
detection, image size: 935, 1241, detect object: person, score: 0.995543, location: x=187, y=43, width=540, height=591
detection, image size: 935, 1241, detect object: person, score: 0.929626, location: x=125, y=639, width=577, height=597

# 获得目标检测结果图片，并查看
adb pull /data/local/tmp/test_ssd_detection_result.jpg ./
```

#### 口罩检测示例

```shell
cd inference_lite_lib.android.armv8/demo/cxx/mask_detection

# 准备预测部署文件
bash prepare.sh

# 执行预测
cd mask_demo && bash run.sh

# 运行成功后，将在控制台输出如下内容，可以打开test_img_result.jpg图片查看预测结果
../mask_demo/: 9 files pushed, 0 skipped. 141.6 MB/s (28652282 bytes in 0.193s)
Load detecion model succeed.
Detecting face succeed.
Load classification model succeed.
detect face, location: x=237, y=107, width=194, height=255, wear mask: 1, prob: 0.987625
detect face, location: x=61, y=238, width=166, height=213, wear mask: 1, prob: 0.925679
detect face, location: x=566, y=176, width=245, height=294, wear mask: 1, prob: 0.550348
write result to file: test_img_result.jpg, success.
/data/local/tmp/mask_demo/test_img_result.jpg: 1 file pulled, 0 skipped. 13.7 MB/s (87742 bytes in 0.006s)
```

## C++ 应用开发说明

C++代码调用Paddle-Lite执行预测库仅需以下五步：

(1) 引用头文件和命名空间

```c++
#include "paddle_api.h"
using namespace paddle::lite_api;
```

(2) 指定模型文件，创建Predictor

```C++
// 1. Set MobileConfig, model_file_path is 
// the path to model model file. 
MobileConfig config;
config.set_model_from_file(model_file_path);
// 2. Create PaddlePredictor by MobileConfig
std::shared_ptr<PaddlePredictor> predictor =
    CreatePaddlePredictor<MobileConfig>(config);
```

(3) 设置模型输入 (下面以全一输入为例)

```c++
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
input_tensor->Resize({1, 3, 224, 224});
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}
```

(4) 执行预测

```c++
predictor->Run();
```

(5) 获得预测结果

```c++
std::unique_ptr<const Tensor> output_tensor(
    std::move(predictor->GetOutput(0)));
// 转化为数据
auto output_data=output_tensor->data<float>();
```

详细的C++ API说明文档位于[C++ API](../api_reference/cxx_api_doc)。更多C++应用预测开发可以参考位于位于[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)的工程示例代码。
