# Paddle Lite FAQ 

### 一般信息

1、 Paddle Lite和Paddle Inference的相关资料：

答：Paddle Lite 文档 [https://paddle-lite.readthedocs.io/zh/latest/](https://paddle-lite.readthedocs.io/zh/latest/)

基于Paddle Lite实现的安卓、iOS、和ARMLinux上的项目[https://github.com/PaddlePaddle/Paddle-Lite-Demo
](https://github.com/PaddlePaddle/Paddle-Lite-Demo)

2、 PaddleSlim与Paddle Lite、Paddle Inference对于量化的支持是怎么样的？
答：
开发者可以通过PaddleSlim获得量化模型。Paddle Lite和Paddle Inferece对int8的量化支持是最为完备，因此推荐飞桨的开发者在实际使用中，使用int8量化。
另外，Paddle Inference还支持fp16的混合精度推理。


### 编译相关

1、把paddle-lite的.a和头文件移到地图的工程里编译ios的demo，出现编译错误

答：请检查xcode的header search path和library search path有没有正确设置

2、在Host端采用交叉编译方式编译Paddle Lite，将编译后的`libpaddle_light_api_shared.so`和可执行程序放到板卡上运行，出现了如下图所示的错误，怎么解决？ 
![host_target_compiling_env_miss_matched](https://user-images.githubusercontent.com/9973393/75761527-31b8b700-5d74-11ea-8a9a-0bc0253ee003.png)
答： 原因是Host端的交叉编译环境与Target端板卡的运行环境不一致，导致libpaddle_light_api_shared.so链接的GLIBC库高于板卡环境的GLIBC库。目前有四种解决办法（为了保证编译环境与官方一致，推荐第一种方式）：1）在Host端，参考[编译环境准备](../source_compile/compile_env)和[Linux源码编译](../source_compile/compile_linux)中的Docker方式重新编译libpaddle_light_api_shared.so；2）在Host端，使用与Target端版本一致的ARM GCC和GLIBC库重新编译libpaddle_light_api_shared.so；3）在Target端板卡上，参考[编译环境准备](../source_compile/compile_env)和[Linux源码编译](../source_compile/compile_linux)中的ARM Linux本地编译方式重新编译libpaddle_light_api_shared.so；4）在Target端板卡上，将GLIBC库升级到和Host端一致的版本，即GLIBC2.27。

### 模型转换

1、使用OPT工具转换失败的一般步骤

答：
首先确认必须要**使用相同版本的opt和预测库**。
如果报错有某个op不支持，那么需要opt新增对该算子的支持。 
使用./opt --print_all_ops=true, 可以查看当前Paddle Lite支持的所有op。另外，也可以尝试编译develop版本的代码，有可能对未支持的op做了支持，但是尚未合入到稳定版本。

2、 什么是combined model，什么是uncombined model？

答：这两类模型都是paddle保存出来的可用于预测的模型，区别在于模型的参数文件给出的形式。
combined model包含两个文件，一个是模型结构的__model__文件，和表示模型参数的params文件。
uncombined model除了模型结构的__model__文件以外，还包含若干的模型参数文件, 如下图所示。
![图片](https://paddlelite-data.bj.bcebos.com/doc_images%2Fseperated_model.png)

3、如果出现找不到 `__model__`文件、version不匹配、加载模型时`segmentation_fault`等问题，怎么办？

答：这时候，请仔细检查模型的输入路径是否正确。如果输入的路径不正确，容易出现以上问题。比如需要注意，combined model和uncombined model的参数输入方式是不同。前者的输入需要通过两个参数`--model_file`，`--param_file`来分别指定模型和参数文件，后者只需要一个参数`--model_file`来指定模型的目录。

4、查看某个版本是否支持当前模型的方法。

答：以yolov3为例，使用OPT工具查看下是否支持，参考命令如下：

```
./lite/api/opt --print_model_ops=true --model_file=$MODEL_FILE --param_file=$PARAM_FILE \

               --valid_targets=arm
```


5、在转换opt的时候报错，需要怎么排查？

答：需要查看是否是combined model，二者使用的参数不同。对于combined model，需要set_model_file(./infer_model/__model__) 和 set_parram_file(./infer_model/__param__) 2个API都需要调用。

6、2.7版本的opt_mac是一个不可执行文件

答：如果遇到此类问题，不推荐使用release的opt_mac工具，可以考虑直接使用pip install paddlelite进行模型转换。

7、DLTP-17809 [自动收录问题] 【AUTO 2020-12-07 Paddle-Lite】bash prepare_demo.bash arm8时报错 
```
tar: Ignoring unknown extended header keyword 'SCHILY.dev'
tar: Ignoring unknown extended header keyword 'SCHILY.ino'
tar: Ignoring unknown extended header keyword 'SCHILY.nlink'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.creationtime'
tar: Ignoring unknown extended header keyword 'SCHILY.dev'
tar: Ignoring unknown extended header keyword 'SCHILY.ino'
tar: Ignoring unknown extended header keyword 'SCHILY.nlink'
tar: Ignoring unknown extended header keyword 'SCHILY.dev'
```
用户遇到此类问题，如何解答？

答：这拉取第三方库失败，建议采用网络代理外部链接提高下载速度。

### 硬件和OS支持

1、Paddle Lite支持英伟达的Jetson硬件吗？

答：目前Paddle Lite只支持透过Jetson硬件上的ARM CPU做推理, 如果有使用TensorRT加速库需求的用户，我们建议使用飞桨的原生推理库[paddle inference](https://paddle-inference.readthedocs.io/en/latest/#)，如果您在使用Jeston硬件时有小于1MB的轻量推理库强需求，请反馈给我们: wangyunkai@baidu.com，我们会视情况加速TensorRT在Paddle Lite上的支持排期

2、Paddle Lite如何支持低版本的安卓？比如低于安卓5.1的系统。相关issue参考：
答：
在编译的时候，使用如下的编译选项 `--android_api_level`。详情可参考[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tools/build_android.sh](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tools/build_android.sh)
安卓版本低于6.0时，需要设置api level。Android 5.1， api level为22， android5.0，api level为21。不支持低于Android 5.0的版本。
```
--android_api_level: (21~27); control android api level, default value is 22 when arch=armv7 , 23 when arch=armv8.               |"
    echo -e "|                 eg. when android platform version is lower than Android6.0, we need to set android_api_level:                        |"
    echo -e "|                     Android5.1: --android_api_level=22    Android5.0: --android_api_level=21     LowerThanAndroid5.0: not supported  |"
```
3、树莓派是否支持飞桨的动态图？动态图的模型如何保存被Paddle lite支持？

答：支持。动态图需要通过paddle.jit.save来保存。对于动态图可参考以下例子来保存模型。
```
    #保存静态图模型, 用于部署
    x_spec = paddle.static.InputSpec(shape=[None, 3, 224, 224], name='img') # 定制化预测模型导出
    model = paddle.jit.to_static(model, input_spec=[x_spec])
    paddle.jit.save(model, "MyCNN")
```
保存下来模型之后，可以正常使用opt工具来将模型转换为.nb的格式。

### 常见报错

1、Check failed: op: no Op found for hard_sigmoid Aborted (core dumped) 遇到该问题怎么办？

答：请在编译的时候打开with_extra=true，试试是否能够解决。

2、Only version 0 is supported，遇到类似报错的检测方法

https://github.com/PaddlePaddle/Paddle-Lite/issues/4763

答： 1. 首先确认模型是通过save_inference_model这个api保存下来的，通过paddle lite预测，需要飞桨的模型是用该api保存下来的。 2.其次检查输入的模型参数是否正确，比如对于combined model，需要指定–model_file，和–model_param两个参数，对于uncombined model，需要指定–model_dir一个参数。
