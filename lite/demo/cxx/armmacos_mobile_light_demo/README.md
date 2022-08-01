# c++ demo on macOS (Apple Silicon based)
该文档介绍在 c++ 示例在 macOS (Apple Silicon based) 平台编译。

## 编译示例
示例默认支持 OpenCL 检测。
```
sh build.sh
```
如果需要编译支持 Metal 的运行示例，需要修改编译脚本```build.sh```，cmake 后添加```-DMETAL=ON```选项
