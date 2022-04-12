# Paddle Lite FAQ 

### 编译相关

1、编译 Paddle Lite 报错怎么办？

答：不推荐用户自己编译，推荐使用预编译库 [Paddle Lite 预编译库下载](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html)

2、遇到 Check failed: op: no Op found for xxx Aborted (core dumped) 问题怎么办？

答：更换当前 Paddle Lite 预测库为带 `with_extra = ON` 标签的预编译库。

3、ARM CPU 端多线程支持情况，某些case下，多线程没有效果？

答：gcc 编译模式下，V7/V8 多线程均支持；clang 编译模式下，V8 支持多线程，V7 只支持单线程

### 模型转换

1、使用 OPT 工具转换模型报错怎么办？

答：首先确认必须要**使用相同版本的 OPT 和预测库**。如果报不支持的 `op`，使用 `./opt --print_all_ops=true`, 可以查看当前 Paddle Lite 支持的所有 `op`。

2、查看某个版本的预测库是否支持当前模型的方法。

答：以 yolov3 为例，使用同版本 OPT 工具参考命令如下：

```
./opt --print_model_ops=true --model_dir=$MODEL_FILE_PATH --valid_targets=arm
```

### 使用问题

1、Paddle Lite 相关资料：

答：Paddle Lite 文档 [Paddle Lite 文档链接](https://paddle-lite.readthedocs.io/zh/latest/)。  
   Paddle Lite 各平台应用示例 [Paddle Lite 应用示例链接](https://github.com/PaddlePaddle/Paddle-Lite-Demo)。

2、Paddle Lite 如何加载动态图模型？

答：动态图需要通过 `paddle.jit.save` 保存成静态图再用 OPT 工具转换。可参考：

```shell
    #保存静态图模型, 用于部署
    x_spec = paddle.static.InputSpec(shape=[None, 3, 224, 224], name='img') 
    # 定制化预测模型导出
    model = paddle.jit.to_static(model, input_spec=[x_spec])
    paddle.jit.save(model, "MyCNN")
```

### 硬件和 OS 支持

1、Paddle Lite 支持英伟达的 Jetson 硬件吗？

答：不支持, 对于英伟达的 Jetson 硬件，推荐用户使用飞桨的原生推理库 [paddle inference](https://paddle-inference.readthedocs.io/en/latest/#)。

2、Paddle Lite 如何支持低版本的安卓？

答：推荐用户下载预编译库，如不满足条件使用，可参考编译脚本 [build_android.sh](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tools/build_android.sh)
进行编译。Android 版本低于6.0时，需要设置 `--android_api_level`，不支持低于 Android 5.0 的版本。
