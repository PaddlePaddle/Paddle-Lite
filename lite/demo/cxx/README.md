# C++ Demo
1. 环境准备
   - 保证Android NDK在/opt目录下
   - 一台armv7或armv8架构的安卓手机
2. 编译并运行全量api的demo(注：当编译模式为tiny_pubish时将不存在该demo)
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

3. 编译并运行轻量级api的demo
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

4. 编译并运行ssd目标检测的demo
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

5. 编译并运行yolov3目标检测的demo
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

6. 编译并运行物体分类的demo
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
    
9. 编译含CV预处理库模型单测demo 
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

10. 编译并运行mask_detection口罩检测的demo

注：运行该demo所需的libpaddle_light_api_shared.so，编译选项需使用build_extra=ON

```shell
cd ../mask_detection
wget https://paddle-inference-dist.bj.bcebos.com/mask_detection.tar.gz
tar zxvf mask_detection.tar.gz
make
adb push mask_detection /data/local/tmp/
adb push test.jpg /data/local/tmp/
adb push face_detection /data/local/tmp
adb push mask_classification /data/local/tmp
adb push ../../../cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/
adb shell chmod +x /data/local/tmp/mask_detection
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/:$LD_LIBRARY_PATH && 
/data/local/tmp/mask_detection /data/local/tmp/face_detection \
/data/local/tmp/mask_classification /data/local/tmp/test.jpg"
adb pull /data/local/tmp/test_mask_detection_result.jpg ./
```
运行成功将在mask_detection目录下看到生成的口罩检测结果图像: test_mask_detection_result.jpg
