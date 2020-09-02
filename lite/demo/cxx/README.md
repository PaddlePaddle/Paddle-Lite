# C++ Demo

> 欢迎加入PaddleLite百度官方QQ群（696965088），会有专业同学解答您的疑问与困惑。

1. 环境准备
   - 一台可以编译PaddleLite的电脑
   - 一台armv7或armv8架构的安卓手机

2. 人脸识别和佩戴口罩判断的Demo

目前，PaddleLite提供了shell端的人脸识别和佩戴口罩判断的Demo，首先基于已经准备好的Demo进行演示，然后介绍如何基于代码编译Demo并执行。

**下载Demo并执行**

下载压缩包[mask_demo](https://paddle-inference-dist.cdn.bcebos.com/PaddleLiteDemo/mask_demo_v2.6.tgz)，解压到本地，其中包括编译好的可执行文件、模型文件、测试图片、PaddleLite 2.6版本动态库。

电脑连接安卓手机，在电脑shell端进入 `mask_demo` 目录。

执行 `sh run.sh`，会将文件push到手机端、执行口罩检测、pull结果图片。

在电脑端查看 `test_img_result.jpg`，即是口罩检测结果。


**编译Demo并执行**

参考[编译环境准备](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)准备编译环境。

执行下面命令，下载PaddleLite代码，切换到2.6版本分支。
```shell
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git fetch origin release/v2.6:release/v2.6 
git checkout release/v2.6
```

进入PaddleLite根目录，编译预测库。
```shell
./lite/tools/build.sh \
    --arm_os=android \
    --arm_abi=armv8 \
    --arm_lang=gcc \
    --android_stl=c++_static \
    --build_extra=ON \
    --with_log=ON \
    full_publish
```

编译完成后，进入Demo编译目录，执行脚本，会编译可执行文件，同时将可执行文件、预测库、模型、图片保存到 `mask_demo` 文件中。
```shell
cd build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/mask_detection
sh prepare.sh
```

当然，大家也可以通过PaddleHub下载人脸检测模型和口罩佩戴判断模型，然后使用 `opt`工具转换，最后替换 `mask_demo` 文件中的模型文件。
```
# 参考[文档](https://github.com/PaddlePaddle/PaddleHub)安装PaddleHub

# 参考[文档](https://www.paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_mobile_mask&en_category=ObjectDetection)安装模型，执行 hub install pyramidbox_lite_mobile_mask==1.3.0

#通过python执行以下代码，将模型保存在test_program文件夹之中，人脸检测和口罩佩戴判断模型分别存储在pyramidbox_lite和mask_detector之中。文件夹中的__model__是模型结构文件，__param__文件是权重文件
import paddlehub as hub
pyramidbox_lite_mobile_mask = hub.Module(name="pyramidbox_lite_mobile_mask")
pyramidbox_lite_mobile_mask.processor.save_inference_model(dirname="test_program")

# 从PaddleHub下载的是预测模型，需要使用PaddleLite提供的 opt 对预测模型进行转换，请参考[模型转换文档](https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/model_optimize_tool/)。
```

电脑连接安卓手机，在电脑shell端进入 `mask_demo` 目录。

执行 `sh run.sh`，会将文件push到手机端、执行口罩检测、pull结果图片。

在电脑端查看 `test_img_result.jpg`，即是口罩检测结果，如下图。

![test_mask_detection_result](https://user-images.githubusercontent.com/7383104/75131866-bae64300-570f-11ea-9cad-17acfaea1cfc.jpg)

注：mask_detetion.cc 中的缩放因子shrink, 检测阈值detect_threshold, 可供自由配置:
   - 缩放因子越大，模型运行速度越慢，检测准确率越高。
   - 检测阈值越高，人脸筛选越严格，检测出的人脸框可能越少。

3. 编译并运行全量api的demo(注：当编译模式为tiny_pubish时将不存在该demo)
```shell
cd inference_lite_lib.android.armv8/demo/cxx/mobile_full
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxvf mobilenet_v1.tar.gz
make
adb push mobilenet_v1 /data/local/tmp/
adb push mobilenetv1_full_api /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_full_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobilenetv1_full_api
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
/data/local/tmp/mobilenetv1_full_api --model_dir=/data/local/tmp/mobilenet_v1 --optimized_model_dir=/data/local/tmp/mobilenet_v1.opt"
```
运行成功将在控制台输出预测结果的前10个类别的预测概率

4. 编译并运行轻量级api的demo
```shell
cd ../mobile_light
make
adb push mobilenetv1_light_api /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobilenetv1_light_api
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
/data/local/tmp/mobilenetv1_light_api /data/local/tmp/mobilenet_v1.opt"
```
运行成功将在控制台输出预测结果的前10个类别的预测概率

5. 编译并运行ssd目标检测的demo
```shell
cd ../ssd_detection
wget https://paddle-inference-dist.bj.bcebos.com/mobilenetv1-ssd.tar.gz
tar zxvf mobilenetv1-ssd.tar.gz
make
adb push ssd_detection /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push mobilenetv1-ssd /data/local/tmp
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/ssd_detection
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
/data/local/tmp/ssd_detection /data/local/tmp/mobilenetv1-ssd /data/local/tmp/test.jpg"
adb pull /data/local/tmp/test_ssd_detection_result.jpg ./
```
运行成功将在ssd_detection目录下看到生成的目标检测结果图像: test_ssd_detection_result.jpg

6. 编译并运行yolov3目标检测的demo
```shell
cd ../yolov3_detection
wget https://paddle-inference-dist.bj.bcebos.com/mobilenetv1-yolov3.tar.gz
tar zxvf mobilenetv1-yolov3.tar.gz
make
adb push yolov3_detection /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push mobilenetv1-yolov3 /data/local/tmp
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/yolov3_detection
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
/data/local/tmp/yolov3_detection /data/local/tmp/mobilenetv1-yolov3 /data/local/tmp/test.jpg"
adb pull /data/local/tmp/test_yolov3_detection_result.jpg ./
```
运行成功将在yolov3_detection目录下看到生成的目标检测结果图像: test_yolov3_detection_result.jpg

7. 编译并运行物体分类的demo
```shell
cd ../mobile_classify
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxvf mobilenet_v1.tar.gz
./model_optimize_tool optimize model
make

adb push mobile_classify /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push labels.txt /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/mobile_classify
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
/data/local/tmp/mobile_classify /data/local/tmp/mobilenetv1opt2 /data/local/tmp/test.jpg /data/local/tmp/labels.txt"
```
运行成功将在控制台输出预测结果的前5个类别的预测概率
- 如若想看前10个类别的预测概率，在运行命令输入topk的值即可
    eg:
    ```shell
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
    /data/local/tmp/mobile_classify /data/local/tmp/mobilenetv1opt2/ /data/local/tmp/test.jpg /data/local/tmp/labels.txt 10"
    ```
- 如若想看其他模型的分类结果， 在运行命令输入model_dir 及其model的输入大小即可
    eg:
    ```shell
    adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
    /data/local/tmp/mobile_classify /data/local/tmp/mobilenetv2opt2/ /data/local/tmp/test.jpg /data/local/tmp/labels.txt 10 224 224"
    ```
    
8. 编译含CV预处理库模型单测demo 
```shell
cd ../test_cv
wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxvf mobilenet_v1.tar.gz
./model_optimize_tool optimize model
make
adb push test_model_cv /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push labels.txt /data/local/tmp/
adb push ../../../cxx/lib/libpaddle_full_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/test_model_cv
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
/data/local/tmp/test_model_cv /data/local/tmp/mobilenetv1opt2 /data/local/tmp/test.jpg /data/local/tmp/labels.txt"
```
运行成功将在控制台输出预测结果的前10个类别的预测概率
