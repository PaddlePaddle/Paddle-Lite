1. 环境准备
   - 一台可以编译PaddleLite的电脑
   - 一台armv7或armv8架构的安卓手机

2. 编译

参考[编译环境准备](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)准备编译环境。

执行下面命令，下载PaddleLite代码，切换到特定版本分支。

```shell
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git fetch origin release/v2.8:release/v2.8
git checkout release/v2.8
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

编译完成后，进入Demo编译目录，执行脚本`prepare.sh`，会编译可执行文件，同时将可执行文件、预测库、模型、数据保存到 `lac_demo_file` 文件中。

```shell
cd build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/demo/cxx/lac_demo
sh prepare.sh
```

3. 执行

电脑连接安卓手机，在电脑shell端进入 `lac_demo_file` 目录。

执行 `sh run.sh`，会将文件push到手机端、执行lac模型、输出预测结果的准确率。
