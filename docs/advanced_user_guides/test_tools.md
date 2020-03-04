# 测试工具

Basic profiler 用于 CPU 上kernel 耗时的统计。

## 开启方法:

参照 [编译安装](../installation/source_compile) 中的**full_publish**部分进行环境配置，在 cmake 时添加 `-DLITE_WITH_PROFILE=ON` ，就可以开启相应支持。

## 使用示例：

在模型执行完毕后，会自动打印类似如下 profiler 的日志

```
                        kernel   average       min       max     count
                feed/def/1/4/2         0         0         0         1
              conv2d/def/4/1/1      1175      1175      1175         1
              conv2d/def/4/1/1      1253      1253      1253         1
    depthwise_conv2d/def/4/1/1       519       519       519         1
              conv2d/def/4/1/1       721       721       721         1
     elementwise_add/def/4/1/1        18        18        18         1
              conv2d/def/4/1/1      2174      2174      2174         1
    depthwise_conv2d/def/4/1/1       380       380       380         1
              conv2d/def/4/1/1       773       773       773         1
     elementwise_add/def/4/1/1         2         2         2         1
              conv2d/def/4/1/1      1248      1248      1248         1
    depthwise_conv2d/def/4/1/1       492       492       492         1
              conv2d/def/4/1/1      1150      1150      1150         1
     elementwise_add/def/4/1/1        33        33        33         1
     elementwise_add/def/4/1/1         3         3         3         1
              conv2d/def/4/1/1      1254      1254      1254         1
    depthwise_conv2d/def/4/1/1       126       126       126         1
```
