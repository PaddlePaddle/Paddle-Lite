# 基于 GoogleBenchmark 的性能测试用例使用方法

## 编译
* 在编译PaddeLite过程中, 执行 cmake 时需要添加`-DLITE_WITH_BENCHMARK_TEST=ON`选项.
* cmake 完成后,需要进入build目录手动 make 相关 target ,例如 `make f32-gemm-bench`
    * 相关的 target 可以在`CMakeLists.txt`文件中查询
* 目前的测试用例仅考虑了ARM平台,如果要支持更多的平台,需要同时修改测试用例和CMakeLists.txt
    * 测试用例`xxx.cc`中, 应当将平台相关的代码替换为平台无关的.
    * CMakeLists.txt中应当将`LITE_WITH_ARM`相关的内容进行修改.
    * `googlebenchmark`库相关的内容不必修改,该库是平台无关的,且总是会从源码编译.

## 运行
* 编译出的二进制文件没有第三方的动态库依赖,可以直接运行

## 开发及扩展
* 如果有较为深度的开发需求,请参考[Google Benchmark 官方文档](https://github.com/google/benchmark)
* 如果仅仅希望按照自定义的参数运行测试.
    * Google Benchmark 需要用户按 0,1,2,3 ... 的顺序来解析输入参数.请先搜索`state.range(0)`这样的代码,以此来确定0,1,2,3的顺序究竟对应了什么参数.
    * 参考下面的代码片段,添加一个自己的测试用例
    * 使用`--benchmark_filter`参数启动benchmark测试程序,仅运行新添加的测试
        * 例如,下面的代码段对应的参数可以为`--benchmark_filter='.*my_convolution_case.*'`


```cpp
// 这段代码可以直接添加到convolution-arm.cc中,其他case可以类比修改.
static void MyConvolutionCase(benchmark::internal::Benchmark* b) {
  b->Args({1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 8000, 2});
  b->Args({1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2, 8000});
}

BENCHMARK_CAPTURE(int8_conv, my_convolution_case, "dbg net")->Apply(MyConvolutionCase)->UseRealTime();

```
