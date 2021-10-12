# 编译选项说明


| 编译选项 |  说明  | 默认值 |
| :-- |  :-- |--: |
| LITE_WITH_LOG |  是否输出日志信息 | ON |
| LITE_WITH_EXCEPTION | 是否在错误发生时抛出异常 | OFF |
| LITE_WITH_TRAIN |  打开[模型训练功能](../demo_guides/cpp_train_demo.html)，支持移动端模型训练 | OFF |
| LITE_BUILD_EXTRA |  编译[全量预测库](library.html)，包含更多算子和模型支持 | OFF |
| LITE_BUILD_TAILOR | 编译时[根据模型裁剪预测库](library_tailoring.html)，缩小预测库大小 | OFF |
| WITH_SYSTEM_BLAS |  编译时强制使用reference BLAS |  OFF |

### 轻量级编译选项

适用于移动端编译，或者对预测库大小有要求的运行环境：

| 编译选项 |  说明  | 默认值 |
| :-- |  :-- | --: |
| LITE_WITH_LIGHT_WEIGHT_FRAMEWORK | 编译移动端轻量级预测框架 | OFF |
| LITE_ON_TINY_PUBLISH |  编译移动端部署库，无第三方库依赖 | OFF |

### 全功能编译选项

适用于服务端编译，或者对预测库大小没有要求的运行环境：

| 编译选项 |  说明  | 默认值 |
| :-- |  :-- | --: |
| LITE_WITH_PROFILE |  编译[Profiler工具](../user_guides/debug.html)，用于CPU上kernel耗时统计 | OFF |
| LITE_WITH_PRECISION_PROFILE |  开启Profiler工具的模型精度分析功能 | OFF |
| WITH_TESTING |  编译Lite单测模块 | OFF |

## 平台相关编译选项

- [使用 Linux x86 构建 / 目标终端为 Android](./compile_android.md)
- [iOS 源码编译](./compile_ios.md)
- [Windows 源码编译](./compile_windows.md)
- [ARM Linux 源码编译](./compile_linux.md)
- [MacOS M1 芯片源码编译](./compile_armmacos.md)
