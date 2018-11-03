# FPGA开发文档

FPGA平台的代码分为V1和V2。其中V1在Xilinx ZCU102 revision 1.0开发板测试Resnet50成功，预测结果正确。以下描述适用于复现V1运行的结果。

## 准备硬件
___

1. 购买Xilinx ZCU102 revision1.0 开发板
2. 另外下载Xilinx ZCU102 Ubuntu[镜像文件](https://www.xilinx.com/member/forms/download/xef.html?filename=Ubuntu_Desktop_Release_2018_1.zip)，并烧录进SD卡。
 * Windowns系统可使用Win32DiskImager
 * Linux系统使用dd命令：dd if=name.img of=/dev/sdb
2. 将SD卡插入电脑，替换分区1中已有的BOOT.BIN、image.ub为[BOOT.BIN、image.ub](http://mms-graph.bj.bcebos.com/paddle-mobile/fpga/files.tar.gz)
3. 将SD卡插入ZCU102开发板，设置板拨码开关为SD卡启动，上电启动Linux系统.
3. 装载驱动：sudo insmod [fpgadrv.ko](http://mms-graph.bj.bcebos.com/paddle-mobile/fpga/files.tar.gz)


## 编译工程
___
1. 将最新的paddle mobile 代码复制到ZCU102开发板中。
2. 进入paddle-mobile根目录， CMakeLists.txt 设置平台为 option(FPGA "fpga support" ON)。CPU和MALI\_GPU选项设置为OFF。设置option(FPGAV1 "fpga v1" ON), option(FPGAV2 "fpga v2" OFF)。
2. 执行以下命令，可在./test/build下生成test-resnet50可执行程序。
    * mkdir build
    * cd build
    * cmake ..
    * make

## 准备模型和数据
___
1. 模型文件放在./test/models/resnet50中。将[\_\_model\_\_](http://mms-graph.bj.bcebos.com/paddle-mobile/fpga/files.tar.gz)文件复制到此文件夹下。
2. 如果不存在，则创建文件夹./test/models/resnet50 和 ./test/images。
3. 另外下载模型[权重文件](http://paddle-imagenet-models.bj.bcebos.com/resnet_50_model.tar),解压后也放在./test/models/resnet50 中。
4. 将数据文件[image_src_float](http://mms-graph.bj.bcebos.com/paddle-mobile/fpga/files.tar.gz)复制到./test/images下。此数据文件对应着标准数据集中的ILSVRC2012_val_00000885.JPEG，分类标签为80， 对应着"black grouse"。

## 运行程序
___
1. 进入./test/build目录。
2. sudo ./test-resnet50
3. 如果于DEBUG选项是否打开，屏幕会输出很多中间打印信息。最终打印出预测分类结果为80。
