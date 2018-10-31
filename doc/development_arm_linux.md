# ARM_LINUX开发文档
目前支持直接在arm_linux平台上编译paddle-mobile

## 以Raspberrypi3为例：
### 执行编译
在paddle-mobile根目录中，执行以下命令：
```
cd tools
/bin/bash build.sh arm_linux googlenet
```
执行完毕后，生成的so位于paddle-mobile/build/release/arm-linux/build目录中，单测可执行文件位于test/build目录中。

### 运行
```
cd ../build/release/arm-linux/build
export LD_LIBRARY_PATH=.
cd ../../../../test/build/
./test-googlenet
```
*注1：如果本地test目录下没有模型的话，会自动下载官方demo模型并解压.*

*注2：因为arm_linux设备算力限制,建议编译时,根据需要指定编译某个模型（如googlenet）或扩大系统的swap交换空间，避免编译时卡死.*

## 其他ARM_LINUX平台

其他的arm_linux平台可以修改 tools/build.sh中的相关编译参数进行编译。可以参考对应平台的编译选项。
特别说明的是Android平台请参考Android开发文档.

