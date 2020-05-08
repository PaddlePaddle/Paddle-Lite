# C++ Demo

## 1. 下载最新版本预测库

预测库下载界面位于[Paddle-Lite官方预编译库](../user_guides/release_lib)，可根据需求选择合适版本。

以**Android-ARMv8架构**为例，可以下载以下版本：


|ARM Version|build_extra|arm_stl|target|下载|
|:-------:|:-----:|:-----:|:-----:|:-------:|
|armv8|OFF|c++_static|tiny_publish|[release/v2.3](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/inference_lite_lib.android.armv8.gcc.c++_static.tiny_publish.tar.gz)|

**解压后内容如下图所示：**

![image](https://paddlelite-data.bj.bcebos.com/doc_images/cxx_demo/1inference_lib.png)

## 2. 转化模型

PaddlePaddle的原生模型需要经过[opt]()工具转化为Paddle-Lite可以支持的naive_buffer格式。

以`mobilenet_v1`模型为例：

（1）下载[mobilenet_v1模型](http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz)后解压：

```shell
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxf mobilenet_v1.tar.gz
```

**如下图所示:**

![image](https://paddlelite-data.bj.bcebos.com/doc_images/cxx_demo/3inference_model.png)

（2）下载[opt工具](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/opt)。放入同一文件夹，终端输入命令转化模型：

```shell
wget https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.3.0/opt
chmod +x opt
./opt --model_dir=./mobilenet_v1 --optimize_out_type=naive_buffer   --optimize_out=./mobilenet_v1_opt
```

**结果如下图所示：**

![image](https://paddlelite-data.bj.bcebos.com/doc_images/cxx_demo/2opt_model.png)



## 3. 编写预测程序

准备好预测库和模型，我们便可以编写程序来执行预测。我们提供涵盖图像分类、目标检测等多种应用场景的C++示例demo可供参考，位于`inference_lite_lib.android.armv8/demo/cxx`。

以mobile net_v1预测为例：`mobile_light`为mobilenet_v1预测示例，可以直接调用。

**示例如下图所示：**

![image](https://paddlelite-data.bj.bcebos.com/doc_images/cxx_demo/4light_demo.png)



## 4. 编译

预测程序需要编译为Android可执行文件。

以mobilenet_v1模型为例，C++示例位于`inference_lite_lib.android.armv8/demo/mobile_light`

```shell
cd inference_lite_lib.android.armv8/demo/mobile_light
```

编译demo

```shell
make
```

**结果如下图所示：**

![image](https://paddlelite-data.bj.bcebos.com/doc_images/cxx_demo/5compile_demo.png)

## 5. 执行预测

通过adb工具将可执行文件推送到手机上执行预测

（1）保证电脑已经安装adb工具，手机以"USB调试"、"文件传输模式"连接到电脑。

``` shell
adb deveices   #查看adb设备是否已被识别
```

**连接如下图所示：**

![image](https://paddlelite-data.bj.bcebos.com/doc_images/cxx_demo/6adb_devices.png)

（2）准备预测库、模型和预测文件

1、将模型、动态库和预测文件放入同一文件夹：

![image](https://paddlelite-data.bj.bcebos.com/doc_images/cxx_demo/7files.png)

**注意**：动态预测库文件位于: `inference_lite_lib.android.armv8/cxx/liblibpaddle_light_api_shared.so`

2、文件推送到手机：

``` shell
chmod +x mobilenetv1_light_api
adb push mobilenet_v1_opt.nb /data/local/tmp
adb push libpaddle_light_api_shared.so /data/local/tmp
adb push mobilenetv1_light_api /data/local/tmp
```
**效果如下图所示：**

![image](https://paddlelite-data.bj.bcebos.com/doc_images/cxx_demo/8push_file.png)

（3）执行预测

```shell
adb shell 'cd /data/local/tmp && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp && mobilenetv1_light_api ./mobilenet_v1_opt.nb'
```
**结果如下图所示：**

![image](https://paddlelite-data.bj.bcebos.com/doc_images/cxx_demo/9result.png)

上图的`Output`为mobilenet_v1模型在全1输入时，得到的预测输出。至此，Paddle-Lite的C++ demo执行完毕。





## 注：如何在代码中使用 API

C++代码调用Paddle-Lite执行预测库仅需以下五步：

（1）引用头文件和命名空间

```c++
#include "paddle_api.h"
using namespace paddle::lite_api;
```

（2）指定模型文件，创建Predictor

```C++
// 1. Set MobileConfig, model_file_path is 
// the path to model model file. 
MobileConfig config;
config.set_model_from_file(model_file_path);
// 2. Create PaddlePredictor by MobileConfig
std::shared_ptr<PaddlePredictor> predictor =
    CreatePaddlePredictor<MobileConfig>(config);
```

（3）设置模型输入 (下面以全一输入为例)

```c++
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
input_tensor->Resize({1, 3, 224, 224});
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}
```

（4）执行预测

```c++
predictor->Run();
```

（5）获得预测结果

```c++
std::unique_ptr<const Tensor> output_tensor(
    std::move(predictor->GetOutput(0)));
// 转化为数据
auto output_data=output_tensor->data<float>();
```





## 其他cxx_demo的编译与预期结果

### Light API Demo

```shell
cd ../mobile_light
make
adb push mobilenetv1_light_api /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobilenetv1_light_api
adb shell "/data/local/tmp/mobilenetv1_light_api --model_dir=/data/local/tmp/mobilenet_v1.opt  "
```


### 图像分类 Demo

```shell
cd ../mobile_classify
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxvf mobilenet_v1.tar.gz
make
adb push mobile_classify /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push labels.txt /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobile_classify
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && /data/local/tmp/mobile_classify /data/local/tmp/mobilenet_v1.opt /data/local/tmp/test.jpg /data/local/tmp/labels.txt"
```

### 目标检测 Demo

```shell
cd ../mobile_detection
wget https://paddle-inference-dist.bj.bcebos.com/mobilenetv1-ssd.tar.gz
tar zxvf mobilenetv1-ssd.tar.gz
make
adb push mobile_detection /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobile_detection
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && /data/local/tmp/mobile_detection /data/local/tmp/mobilenetv1-ssd /data/local/tmp/test.jpg"
adb pull /data/local/tmp/test_detection_result.jpg ./
```

### light API Demo 运行结果

运行成功后 ，将在控制台输出预测结果的前10个类别的预测概率：

```shell
Output dim: 1000
Output[0]: 0.000191
Output[100]: 0.000160
Output[200]: 0.000264
Output[300]: 0.000211
Output[400]: 0.001032
Output[500]: 0.000110
Output[600]: 0.004829
Output[700]: 0.001845
Output[800]: 0.000202
Output[900]: 0.000586
```

### 图像分类 Demo 运行结果

运行成功后 ，将在控制台输出预测结果的前5个类别的类型索引、名字和预测概率：

```shell
parameter:  model_dir, image_path and label_file are necessary
parameter:  topk, input_width,  input_height, are optional
i: 0, index: 285, name:  Egyptian cat, score: 0.482870
i: 1, index: 281, name:  tabby, tabby cat, score: 0.471593
i: 2, index: 282, name:  tiger cat, score: 0.039779
i: 3, index: 287, name:  lynx, catamount, score: 0.002430
i: 4, index: 722, name:  ping-pong ball, score: 0.000508
```

### 目标检测 Demo 运行结果

运行成功后 ，将在控制台输出检测目标的类型、预测概率和坐标：

```shell
running result:
detection image size: 935, 1241, detect object: person, score: 0.996098, location: x=187, y=43, width=540, height=592
detection image size: 935, 1241, detect object: person, score: 0.935293, location: x=123, y=639, width=579, height=597
```
