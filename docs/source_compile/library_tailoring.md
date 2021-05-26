
# 裁剪预测库

Paddle-Lite支持**根据模型裁剪预测库**功能。Paddle-Lite的一般编译会将所有已注册的operator打包到预测库中，造成库文件体积膨胀；**裁剪预测库**能针对具体的模型，只打包优化后该模型需要的operator，有效降低预测库文件大小。

## 效果展示(Android动态预测库体积)

| 测试模型 | 裁剪开关  | **libpaddle_light_api_shared.so** |转化后模型中的OP|
| ------------------ | ---------------------------- | -------- |------------------|
| mobilenetv1（armv8） | 裁剪前 | 1.5 MB       | conv2d,depthwise_conv2d,fc,pool2d,softmax |
| mobilenetv1（armv8） | 裁剪后 |  859 KB     | conv2d,depthwise_conv2d,fc,pool2d,softmax |
| mobilenetv1（armv7） | 裁剪前 | 967 KB | conv2d,depthwise_conv2d,fc,pool2d,softmax |
| mobilenetv1（armv7） | 裁剪后 | 563 KB |conv2d,depthwise_conv2d,fc,pool2d,softmax|


## 实现过程：


### Step 1. 准备模型
- 模型格式：只支持以下五种模型格式
``` shell
# 格式一 : __model__ + var1 + var2 + ...
# 格式二 : model + var1 + var2 + ...
# 格式三 : pdmodel + pdiparams
# 格式四 : model +  params
# 格式五 : model + weights
```

- 所有模型放入同一个文件夹

```bash
# eg. 下面将mobilenet_v1和shufflenet_v1 两个模型放入同一个文件夹 models
# 假设models 文件夹的绝对路径是 /models
/models
 ｜- mobilenet_v1
 ｜       ｜-- model
 ｜       ｜-- params
 ｜- shufflenet_v1
          |-- __model__
          |-- var1
          |-- var1
          |-- ...
```
### Step 2-1. 编译Android 预测库
- 根据模型编译

``` shell
cd Paddle-Lite 
./lite/tools/build_android_by_models.sh /models
# “模型文件夹的绝对路径” 作为脚本输入
```

- 编译产出

```shell
# 编译产出位于： Paddle-Lite/android-lib
android_lib  (Android 编译产出)
   |---- armv7.clang      （armv7 clang预测库&demo)
   |---- armv8.clang      （armv8 clang预测库&demo)
   |---- opt              （模型转换工具opt)
   |---- optimized_model  （opt转化后的Android移动端模型)
              |---- mobilenet_v1.nb
              |---- shufflenet_v1.nb
```

- 其他： 可以修改   `build_android_by_models.sh` 以改变编译选项

``` shell
# Paddle-Lite/lite/tools/build_android_by_models.sh

  8 WITH_LOG=OFF      # （1）可以修改 ON：运行时输出日志  OFF： 运行时不输出日志
  9 WITH_CV=ON        # （2）可以修改 ON：包含图像处理API OFF：不含图像处理API
 10 WITH_EXCEPTION=ON # （3）可以修改 ON：DEBUG选项（可回溯错误信息）
 11 TOOL_CHAIN=clang  #  (4) DNK 编译器： 可选择 clang 或着 gcc
```
### Step 2-2. 编译iOS 预测库

- 根据模型编译

``` shell
cd Paddle-Lite 
./lite/tools/build_ios_by_models.sh /models
# “模型文件夹的绝对路径” 作为脚本输入
```

- 编译产出

```shell
# 编译产出位于： Paddle-Lite/iOS-lib
iOS_lib  (Android 编译产出)
   |---- armv7            （armv7 iOS预测库&demo)
   |---- armv8            （armv8 iOS预测库&demo)
   |---- opt              （模型转换工具opt)
   |---- optimized_model  （opt转化后的iOS移动端模型)
              |---- mobilenet_v1.nb
              |---- shufflenet_v1.nb
```

- 其他： 可以修改   `build_ios_by_models.sh` 以改变编译选项

``` shell
# Paddle-Lite/lite/tools/build_ios_by_models.sh

  8 WITH_LOG=OFF      # （1）可以修改 ON：运行时输出日志  OFF： 运行时不输出日志
  9 WITH_CV=ON        # （2）可以修改 ON：包含图像处理API OFF：不含图像处理API
 10 WITH_EXCEPTION=ON # （3）可以修改 ON：DEBUG选项（可回溯错误信息）
```
