# 源码编译 (Windows)

**注意：** 以下编译方法只适用于release/v2.6.0及之后版本(包括 v2.6.0)。release/v2.3及之前版本(包括 v2.3)请参考[release/v2.3源码编译方法](v2.3_compile.md)。

如果您还没有配置好Windows编译环境，请先根据[编译环境准备](compile_env)中的内容，根据您的开发环境安装编译Windows预测库所需的编译环境。

### 编译

1、 下载代码
```bash
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
# 切换到release分支，比如v2.9
git checkout v2.9
```
2、 源码编译(需要按照提示输入对应的参数)

```dos
cd Paddle-Lite
lite\tools\build_windows.bat with_extra with_profile

# 注意默认编译Windows x64平台，如果需要编译x86平台，需要加入build_x86选项
lite\tools\build_windows.bat build_x86

# 如果需要根据模型裁剪预测库，则需要在with_strip之后输入opt model dir的路径
lite\tools\build_windows.bat with_strip D:\Paddle-Lite\opt_model_dir
```

编译脚本`build_windows.bat`，追加参数说明：

|   参数     |     介绍     |     值     |
|-----------|-------------|-------------|
|  without_log | 可选，是否编译带日志的预测库（默认为ON，即日志打开）| `ON`、`OFF` |
|  without_python | 可选，是否编译python预测库（默认为ON，即编译Python）| `ON`、`OFF` |
|  with_extra | 可选，是否编译全量预测库（当Python打开时默认打开，否则为OFF)，详情可参考[预测库说明](./library.html)。 | `ON`、`OFF` |
|  with_profile | 可选，是否支持逐层耗时分析（默认为OFF）| `ON`、`OFF` |
|  with_precision_profile | 可选，是否支持逐层精度分析（默认为OFF）| `ON`、`OFF` |
|  with_strip | 可选，是否根据模型裁剪预测库（默认为OFF），详情可参考[裁剪预测库](../source_compile/library_trailoring.html)。| `ON`、`OFF` |
|  build_x86 | 可选，是否编译X86平台预测库（默认为OFF，即编译X64平台）| `ON`、`OFF` |
|  with_static_mkl | 可选，是否静态链接Intel(R) MKL加速库（默认为OFF，即动态链接) | `ON`、`OFF` |
|  with_dynamic_crt | 可选，是否动态链接MSVC Rumtime即MD_DynamicRelease（默认为OFF，即静态链接) | `ON`、`OFF` |
|  with_opencl | 可选，是否开启OpenCL（默认为OFF，即编译的预测库仅在CPU上运行，当设为`ON`时，编译出的预测库支持在GPU上运行) | `ON`、`OFF` |
|  without_avx | 可选，使用AVX/SSE指令对x86 Kernel进行加速（默认为ON) | `ON`、`OFF` |

### 编译结果说明

x86编译结果位于 `build.lite.x86/inference_lite_lib`

**具体内容**说明：

1、 `cxx`文件夹：包含c++的库文件与相应的头文件

- `include`  : 头文件
- `lib` : 库文件
  - 静态库文件：
    - `libpaddle_api_full_bundled.lib`  ：full_api 静态库
    - `libpaddle_api_light_bundled.lib` ：light_api 静态库

2、 `third_party` 文件夹：依赖的第三方预测库mklml

- mklml : Paddle-Lite预测库依赖的mklml数学库

3、 `demo/cxx`文件夹：x86预测库的C++ 示例demo

- `mobilenetv1_full` ：使用full_api 执行mobilenet_v1预测的C++ demo
- `mobilenetv1_light` ：使用light_api 执行mobilenet_v1预测的C++ demo

4、 `demo/python`: x86预测库的Python示例demo

- `mobilenetv1_full_api.py`:使用full_api 执行mobilenet_v1预测的Python demo
- `mobilenetv1_light_api.py`:使用full_api 执行mobilenet_v1预测的Python demo

5、 `python`文件夹：包含python的库文件和对应的.whl包

- `install`文件夹：编译成功的.whl包位于`install/dist/*.whl`
- `lib`文件夹：.whl包依赖的库文件
