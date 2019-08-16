# Java Android Demo

要编译和跑起 ./android 文件夹下的 Android demo 程序 PaddlePredictor，你需要准备：

1. 一台能运行安卓程序的安卓手机
2. 一台带有AndroidStudio的开发机

## 编译

首先在PaddleLite的开发Docker镜像中，拉取最新PaddleLite代码，编译对应你手机架构的预测库，
下面我们以arm8 架构举例。进入paddlelite 目录，运行以下cmake 和make 命令：

```
mkdir -p build.lite.android.arm8.gcc
cd build.lite.android.arm8.gcc

cmake .. \
-DWITH_GPU=OFF \
-DWITH_MKL=OFF \
-DWITH_LITE=ON \
-DLITE_WITH_JAVA=ON \
-DLITE_WITH_CUDA=OFF \
-DLITE_WITH_X86=OFF \
-DLITE_WITH_ARM=ON \
-DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
-DWITH_TESTING=OFF \
-DLITE_SHUTDOWN_LOG=ON \
-DLITE_ON_TINY_PUBLISH=ON \
-DARM_TARGET_OS=android -DARM_TARGET_ARCH_ABI=armv8 -DARM_TARGET_LANG=gcc

make publish_inference -j4
```

Make完成后查看要存在
```
build.lite.android.arm8.gcc/lite/api/android/jni/native/libpaddle_lite_jni.so
build.lite.android.arm8.gcc/lite/api/android/jni/PaddlePredictor.jar
```
这两个文件。他们分别为 PaddleLite c++ 动态链接库和 Java jar 包。包含 PaddleLite Java API，接下来 Android Java 代
码会使用这些api 

## 准备 demo 需要的其他文件

Demo 除了代码，还需要准备 JNI .so 库（上节提到的`libpaddle_lite_jni.so`），Java .jar 包（上文提到的
`PaddlePredictor.jar` ），和模型文件。我们提供了自动化的脚本和手动拷贝两种方法，用户可以根据自己需要选择：

### 脚本方法

进入 `build.lite.android.armv8/inference_lite_lib.android.armv8/demo/java/android/`，我们准备了
一个脚本`prepare_demo.bash`，脚本输入一个参数，为你要拷贝的.so 对应的架构文件夹名。

例如运行
```
bash prepare_demo.bash armv8
```
该脚本自动下载并解压缩模型文件，拷贝了 .jar 包进demo，还有生成的.so包进  `PaddlePredictor/app/src/main/jinLibs/架构文件夹下`，
在我们这个例子里，armv8 就是架构文件夹。备注：这种方式构建的 demo 在 armv8 手机运行正常。如果要 demo 程序
在别的手机架构（如 armv7）上也运行正常，需要添加别的架构。

### 手动拷贝方法

接下来我们介绍手动拷贝，如果使用了脚本，那么可以跳过以下手动方法的介绍。

### 把 .so 动态库和 .jar 拷贝进安卓demo程序：

把本文件夹下 demo/PaddlePredictor 载入到AndroidStudio。把上一步提到的`libpaddle_lite_jni.so`
拷贝进 `PaddlePredictor/app/src/main/jinLibs/架构文件夹下` 比如文件夹arm8里要包含该 .so文件：
把上一步提到的 `PaddlePredictor.jar` 拷贝进 `PaddlePredictor/app/libs` 下

### 把demo使用到的模型文件拷贝进安卓程序：

下载我们的5个模型文件，并解压缩到 `PaddlePredictor/app/src/main/assets` 这个文件夹中
需要拷贝的模型文件和下载地址：

    inception_v4_simple_opt.nb http://paddle-inference-dist.bj.bcebos.com/inception_v4_simple_opt.nb.tar.gz
    lite_naive_model_opt.nb    http://paddle-inference-dist.bj.bcebos.com/lite_naive_model_opt.nb.tar.gz
    mobilenet_v1_opt.nb        http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1_opt.nb.tar.gz
    mobilenet_v2_relu_opt.nb   http://paddle-inference-dist.bj.bcebos.com/mobilenet_v2_relu_opt.nb.tar.gz
    resnet50_opt.nb            http://paddle-inference-dist.bj.bcebos.com/resnet50_opt.nb.tar.gz

下载完后，assets文件夹里要包含解压后的上面五个模型文件夹，但demo里不需要保存原压缩.tar.gz 文件。

## 运行 Android 程序结果

以上准备工作完成，就可以开始Build ，安装，和跑安卓demo程序。当你运行PaddlePredictor 程序时，大概会等10秒，
然后看到类似以下字样：

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

该 demo 程序跑我们的 5 个模型，第一个模型结果将真正的头两个数字输出，并在第二行附上期望的正确值。你应该要
看到他们的误差小于0.001。后面四个模型如果你看到 test:true 字样，说明模型输出通过了我们在 demo 程序里对其输出
的测试。time 代表该测试花费的时间。 

## Android demo 程序的 Instrumented Test 

本节对于想通过命令行自动化demo程序的测试人员

要通过命令行运行demo程序在手机上，进入 demo 的 `PaddlePredictor` 文件夹，运行
```
./gradlew init
```
以上命令只要运行一次，其初始化demo能运行的任务。之后可以通过以下命令运行我们的测试
```
./gradlew connectedAndroidTest
```
