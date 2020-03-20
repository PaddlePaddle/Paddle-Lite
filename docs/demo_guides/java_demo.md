# Java Demo

本节中，Java demo 完整代码位于 [demo/java](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/java) 。

要编译和跑起Android demo 程序 PaddlePredictor，你需要准备：

1. 一台能运行安卓程序的安卓手机
2. 一台带有AndroidStudio的开发机

## 编译

首先在PaddleLite的开发 [Docker镜像](../user_guides/source_compile) 中，拉取最新PaddleLite代码，编译对应你手机架构的预测库，
下面我们以arm8 架构举例。进入paddlelite 目录，运行以下命令：

```shell
./lite/tools/build.sh        \
    --arm_os=android         \
    --arm_abi=armv8          \
    --arm_lang=gcc           \
    --android_stl=c++_static \
    tiny_publish
```

命令完成后查看要存在

```
./build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/java/so/libpaddle_lite_jni.so
./build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/java/jar/PaddlePredictor.jar
./build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/java/android
```

libpaddle_lite_jni.so为 PaddleLite c++ 动态链接库，PaddlePredictor.jar为 Java jar 包，两者包含 PaddleLite Java API，接下来 Android Java 代码会使用这些api。android文件夹中则是Android demo。

## 准备 demo 需要的其他文件

Demo 除了代码，还需要准备在Android工程目录下配置好JNI .so 库（上节提到的`libpaddle_lite_jni.so`），Java .jar 包（上文提到的`PaddlePredictor.jar` ），和模型文件。我们提供了自动化的脚本和手动拷贝两种方法，用户可以根据自己需要选择：

### 脚本方法

进入 `build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/java/android`，我们准备了一个脚本`prepare_demo.bash`，脚本输入一个参数，为你要拷贝的.so 对应的架构文件夹名。

例如运行

```
bash prepare_demo.bash arm8
```

该脚本自动下载并解压缩模型文件，拷贝了 .jar 包进demo，还有生成的.so包进`PaddlePredictor/app/src/main/jinLibs/架构文件夹下`，
在我们这个例子里，armv8 就是架构文件夹。备注：这种方式构建的 demo 在 armv8 手机运行正常。如果要demo 程序在别的手机架构（如 armv7）上也运行正常，需要添加别的架构。

### 手动拷贝方法

接下来我们介绍手动拷贝，如果使用了脚本，那么可以跳过以下手动方法的介绍。

### 把 .so 动态库和 .jar 拷贝进安卓demo程序：

1. 将PaddlePredictor 载入到AndroidStudio。
2. 将`libpaddle_lite_jni.so`拷贝进 `PaddlePredictor/app/src/main/jinLibs/架构文件夹下` ，比如文件夹arm8里要包含该 .so文件。
3. 将 `PaddlePredictor.jar` 拷贝进 `PaddlePredictor/app/libs` 下

### 把demo使用到的模型文件拷贝进安卓程序：

下载我们的5个模型文件，并解压缩到 `PaddlePredictor/app/src/main/assets` 这个文件夹中
需要拷贝的模型文件和下载地址：

```
inception_v4_simple_opt.nb http://paddle-inference-dist.bj.bcebos.com/inception_v4_simple_opt.nb.tar.gz
lite_naive_model_opt.nb    http://paddle-inference-dist.bj.bcebos.com/lite_naive_model_opt.nb.tar.gz
mobilenet_v1_opt.nb        http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1_opt.nb.tar.gz
mobilenet_v2_relu_opt.nb   http://paddle-inference-dist.bj.bcebos.com/mobilenet_v2_relu_opt.nb.tar.gz
resnet50_opt.nb            http://paddle-inference-dist.bj.bcebos.com/resnet50_opt.nb.tar.gz
```

下载完后，assets文件夹里要包含解压后的上面五个模型文件夹，但demo里不需要保存原压缩.tar.gz 文件。

注意：输入的模型要求为naive buffer存储格式，您可以通过 [**Model Optimize Tool**](../user_guides/model_optimize_tool) 将fluid模型转为naive buffer存储格式。

## 运行 Android 程序结果

以上准备工作完成，就可以开始Build 、安装、和运行安卓demo程序。当你运行PaddlePredictor 程序时，大概会等10秒，然后看到类似以下字样：

```
lite_naive_model output: 50.213173, -28.872887
expected: 50.2132, -28.8729

inception_v4_simple test:true
time: xxx ms

resnet50 test:true
time: xxx ms

mobilenet_v1 test:true
time: xxx ms

mobilenet_v2 test:true
time: xxx ms
```

该 demo 程序跑我们的 5 个模型，第一个模型结果将真正的头两个数字输出，并在第二行附上期望的正确值。你应该要看到他们的误差小于0.001。后面四个模型如果你看到 `test:true` 字样，说明模型输出通过了我们在 demo 程序里对其输出的测试。time 代表该测试花费的时间。
