## paddle-mobile GPU开发文档

编译环境配置方法请参考`development_android.md`文档

1. 下载 paddle-mobile

```shell
git clone https://github.com/PaddlePaddle/paddle-mobile.git

adb pull /system/vendor/lib/libOpenCL.so paddle-mobile/third_party/opencl

# 修改paddle-mobile/CMakeLists.txt文件，执行如下操作:
# option(GPU_CL "opencl gpu" OFF)->option(GPU_CL "opencl gpu" ON)

cd paddle-mobile/tools
sh build.sh android
```

2. 将单测可执行文件和模型部署到手机

执行下面的脚本，该脚本会下载测试需要的 [mobilenet和test_image_1x3x224x224_float(预处理过的 NCHW 文件) 文件](http://mms-graph.bj.bcebos.com/paddle-mobile/opencl_test_src.zip)，在项目下的`test`目录创建模型>和图片文件夹，并将`mobilenet`复制到`paddle-mobile/test/models`目录下，将`test_image_1x3x224x224_float`复制到`paddle-mobile/test/images`目录下

```shell
cd tools
sh ./prepare_images_and_models.sh
```

执行下面命令将可执行文件和预测需要的文件部署到手机

```shell
cd ../tools/android-debug-script
sh push2android.sh
```

3. 在`adb shell`中执行对应的可执行文件（目前只支持mobilenet，后续会支持更多的网络模型）

```shell
adb shell
cd /data/local/tmp/bin/
export LD_LIBRARY_PATH=.
./test-mobilenetgpu
```

4. mobilenet cpu模型预测结果

执行下面命令进行mobilenet cpu的预测

```shell
adb shell
cd /data/local/tmp/bin/
export LD_LIBRARY_PATH=.
./test-mobilenet
```

5. 预测结果

  手机型号：小米6(CPU 835,GPU Adreno 540)

  mobilenet gpu：预测性能，耗时41ms左右。

  mobilenet cpu:

  1线程：108ms
  2线程：65ms
  4线程：38ms

  手机型号：OPPO Findx(CPU 845,GPU Adreno 630)

  mobilenet gpu：预测性能，耗时27ms左右。

  mobilenet cpu:

  1线程：90ms
  2线程：50ms
  4线程：29ms
  
 备注: GPU 在打开log之后, 会大幅增加性能开销,测试benchmark请关闭CmakeList中Log选项
