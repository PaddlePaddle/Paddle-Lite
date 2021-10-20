
# Profiler 工具

Profiler 在 Lite 里分为性能 Profiler 和 精度 Profiler：

- 性能 Profiler ：用于逐层耗时统计，可以获取到模型逐层 ARM CPU / X86 CPU / OpenCL 上 kernel 耗时信息。定位耗时潜在问题；
- 精度 Profiler ：用于逐层精度统计，可以获取到模型逐层每个 Op 的输出 tensor 精度信息。


## 性能 Profiler
### 开启方式
在编译 full_publish 预测库时，加入编译选项`--with_profile=ON`. 例如：

针对 Android 平台，按照下面方式编译：

```shell
./lite/tools/build_android.sh \
  --arch=armv8 \
  --toolchain=clang \
  --android_stl=c++_static \
  --with_profile=ON \
  full_publish
```

针对 x86 Linux 平台，按照下面方式编译：

```shell
./lite/tools/build_linux.sh \
  --with_profile=ON \
  full_publish
```

其它平台，参照文档对各平台的编译方式进行编译，同时加上`--with_profile=ON`选项。

### 输出数据解读

上面方式编译好了之后，进入`build.lite.android.armv8.clang/inference_lite_lib.android.armv8/demo/cxx/mobile_light/`，执行`make`之后产生可执行文件`mobilenetv1_light_api`，将此可执行文件和`build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so`通过 ADB shell 发送到手机的同一目录，同时把模型如`mobilenet_v1.nb`也发送到手机上此目录，执行`./mobilenetv1_light_api ./mobilenet_v1.nb`，会自动打印类似如下三部分日志：

1. Detailed Dispatch Profiler Summary：单次推理的逐 OP 底层 Kernel 层运行耗时，即在`KernelBase::Run()`的前后统计耗时。会排除第一次的计时（因为第一次不准确相当于 wamrup ），若多次执行也会打印多次；
2. Concise Create Profiler Summary：汇总统计的创建 Op 的耗时，即从`Instruction::Run()`开始到`KernelBase::Run()`执行前。会排除掉前 10 次推理；
3. Concise Dispatch Profiler Summary：汇总统计的运行 Op 的耗时，即在`KernelBase::Run()`的前后统计耗时，为 Lite 具体设备的底层 Kernel 层完整耗时，会排除掉前 10 次推理。

上述命令默认会推理 100 次，最后一次的`Detailed Dispatch Profiler Summary`如下所示。

```shell
===== Detailed Dispatch Profiler Summary: N/A, Exclude 1 warm-ups =====
OperatorType         KerneAttr(Place)               KernelFuncName           Remark                     InDim           FilterDim       OutDim          Avg(ms) Min(ms) Max(ms) Last(ms) Avg(%)  GOPs    GOPS
conv2d               arm/float/NCHW                 conv_3x3s2_direct_fp32   3x3p1s2g1d1BiasRelu        1x3x224x224     32x3x3x3        1x32x112x112    0.735   0.709   0.865   0.718   2.60%    0.022   29.50
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g32d1BiasRelu       1x32x112x112    32x1x3x3        1x32x112x112    0.318   0.308   0.370   0.314   1.13%    0.007   22.69
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x32x112x112    64x32x1x1       1x64x112x112    1.370   1.347   1.445   1.368   4.85%    0.051   37.51
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s2g64d1BiasRelu       1x64x112x112    64x1x3x3        1x64x56x56      0.299   0.272   0.364   0.315   1.06%    0.004   12.08
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x64x56x56      128x64x1x1      1x128x56x56     1.256   1.241   1.315   1.244   4.44%    0.051   40.92
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g128d1BiasRelu      1x128x56x56     128x1x3x3       1x128x56x56     0.297   0.290   0.325   0.292   1.05%    0.007   24.33
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x128x56x56     128x128x1x1     1x128x56x56     2.436   2.410   2.526   2.433   8.62%    0.103   42.19
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s2g128d1BiasRelu      1x128x56x56     128x1x3x3       1x128x28x28     0.120   0.117   0.137   0.119   0.42%    0.002   15.05
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x128x28x28     256x128x1x1     1x256x28x28     1.218   1.195   1.255   1.211   4.31%    0.051   42.20
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g256d1BiasRelu      1x256x28x28     256x1x3x3       1x256x28x28     0.151   0.147   0.189   0.149   0.53%    0.004   23.90
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x256x28x28     256x256x1x1     1x256x28x28     2.371   2.343   2.408   2.384   8.39%    0.103   43.34
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s2g256d1BiasRelu      1x256x28x28     256x1x3x3       1x256x14x14     0.071   0.069   0.076   0.070   0.25%    0.001   12.80
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x256x14x14     512x256x1x1     1x512x14x14     1.207   1.174   1.229   1.211   4.27%    0.051   42.57
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.103   0.101   0.130   0.101   0.36%    0.002   17.61
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     2.358   2.313   2.389   2.351   8.34%    0.103   43.59
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.103   0.101   0.116   0.109   0.36%    0.002   17.54
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     2.351   2.317   2.383   2.347   8.32%    0.103   43.72
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.103   0.101   0.118   0.102   0.36%    0.002   17.53
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     2.360   2.316   2.401   2.357   8.35%    0.103   43.55
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.103   0.101   0.112   0.103   0.36%    0.002   17.57
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     2.356   2.316   2.392   2.352   8.33%    0.103   43.62
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.103   0.101   0.116   0.101   0.36%    0.002   17.58
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     2.361   2.321   2.400   2.360   8.35%    0.103   43.52
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s2g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x7x7       0.048   0.046   0.083   0.047   0.17%    0.000   9.48
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x7x7       1024x512x1x1    1x1024x7x7      1.263   1.187   1.294   1.278   4.47%    0.051   40.69
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g1024d1BiasRelu     1x1024x7x7      1024x1x3x3      1x1024x7x7      0.060   0.059   0.068   0.060   0.21%    0.001   14.93
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x1024x7x7      1024x1024x1x1   1x1024x7x7      2.441   2.359   2.474   2.439   8.63%    0.103   42.10
pool2d               arm/float/NCHW                 NotImpl                  globalavgEXPLICIT          1x1024x7x7      N/A             1x1024x1x1      0.014   0.012   0.029   0.013   0.05%    0.000   3.63
fc                   arm/float/NCHW                 NotImpl                  Bias                       1x1024x1x1      1024x1000       1x1000          0.287   0.271   0.324   0.293   1.01%    0.003   10.71
softmax              arm/float/NCHW                 NotImpl                  axis-1                     1x1000          N/A             1x1000          0.006   0.005   0.008   0.006   0.02%    0.000   0.95
```

`Concise Create Profiler Summary`和`Concise Dispatch Profiler Summary`只会打印 1 次，如下所示。

```shell
[I 10/13 11: 1:19. 51 .../addpass/Paddle-Lite/lite/core/program.h:216 ~RuntimeProgram]
Timing cycle = 100
===== Concise Create Profiler Summary: N/A, Exclude 10 warm-ups =====
OperatorType         KerneAttr(Place)               KernelFuncName           Avg(ms) Min(ms) Max(ms) Avg(%)  GOPs    CalledTimes
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      0.015   0.004   0.071   49.91%    1.079   13
conv2d               arm/float/NCHW                 conv_3x3s2_direct_fp32   0.001   0.001   0.002   4.64%    0.022   1
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  0.010   0.000   0.018   31.75%    0.035   13
fc                   arm/float/NCHW                 NotImpl                  0.002   0.001   0.011   5.57%    0.003   1
pool2d               arm/float/NCHW                 NotImpl                  0.002   0.001   0.002   5.16%    0.000   1
softmax              arm/float/NCHW                 NotImpl                  0.001   0.000   0.002   2.97%    0.000   1

[I 10/13 11: 1:19. 51 .../addpass/Paddle-Lite/lite/core/program.h:217 ~RuntimeProgram]
Timing cycle = 100
===== Concise Dispatch Profiler Summary: N/A, Exclude 10 warm-ups =====
OperatorType         KerneAttr(Place)               KernelFuncName           Avg(ms) Min(ms) Max(ms) Avg(%)  GOPs    CalledTimes
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      25.381  24.985  25.896  89.79%    1.079   13
conv2d               arm/float/NCHW                 conv_3x3s2_direct_fp32   0.727   0.709   0.793   2.57%    0.022   1
depthwise_conv2d     arm/float/NCHW                 conv_depthwise_3x3_fp32  1.855   1.810   2.160   6.56%    0.035   13
fc                   arm/float/NCHW                 NotImpl                  0.285   0.277   0.318   1.01%    0.003   1
pool2d               arm/float/NCHW                 NotImpl                  0.013   0.012   0.015   0.05%    0.000   1
softmax              arm/float/NCHW                 NotImpl                  0.006   0.005   0.008   0.02%    0.000   1
```

上面是 Android 端 Arm CPU 的性能 Profiler 结果，根据 KernelFuncName 耗时百分占比，可以进一步分析潜在性能问题。


## 精度 Profiler
### 开启方式
在编译 full_publish 预测库时，加入编译选项`--with_precision_profile=ON`. 例如：

针对移动端 Android 平台，按照下面方式编译：

```shell
./lite/tools/build_android.sh \
  --with_precision_profile=ON \
  full_publish
```

针对 x86 Linux 平台，按照下面方式编译：

```shell
./lite/tools/build_linux.sh \
  --with_precision_profile=ON \
  full_publish
```

其它平台，参照文档对各平台的编译方式进行编译，同时加上`--with_precision_profile=ON`选项。

### 输出数据解读

上面方式编译好了之后，进入`build.lite.android.armv8.clang/inference_lite_lib.android.armv8/demo/cxx/mobile_light/`，执行`make`之后产生可执行文件`mobilenetv1_light_api`，将此可执行文件和`build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/lib/libpaddle_light_api_shared.so`通过 ADB shell 发送到手机的同一目录，同时把模型如`mobilenet_v1.nb`也发送到手机上此目录，执行`./mobilenetv1_light_api ./mobilenet_v1.nb`，会自动打印类似如下日志:

```shell
========================================= Detailed Precision Profiler Summary =========================================
operator:(kernel_info)                        output_tensor_name:(tensor_info)                                       dims            mean            std_deviation   ave_grow_rate*
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_1:arm/float/NCHW                                    {1,32,112,112}  0.446896        0.518169        19.234488
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_1:arm/float/NCHW                                    {1,32,112,112}  0.570640        0.665590        209.341221
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_2:arm/float/NCHW                                    {1,64,112,112}  0.316004        0.304760        88.818940
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_2:arm/float/NCHW                                    {1,64,56,56}    0.324968        0.447986        90.012883
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_3:arm/float/NCHW                                    {1,128,56,56}   0.272442        0.228168        43.343078
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_3:arm/float/NCHW                                    {1,128,56,56}   0.304168        0.306793        225.627028
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_4:arm/float/NCHW                                    {1,128,56,56}   0.141191        0.187449        175.758147
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_4:arm/float/NCHW                                    {1,128,28,28}   0.345938        0.350929        83.093575
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_5:arm/float/NCHW                                    {1,256,28,28}   0.194089        0.175237        58.395946
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_5:arm/float/NCHW                                    {1,256,28,28}   0.163017        0.249815        369.975693
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_6:arm/float/NCHW                                    {1,256,28,28}   0.109570        0.157070        126.682379
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_6:arm/float/NCHW                                    {1,256,14,14}   0.288670        0.379004        326.557287
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_7:arm/float/NCHW                                    {1,512,14,14}   0.161610        0.194230        108.015092
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_7:arm/float/NCHW                                    {1,512,14,14}   0.137904        0.217275        580.738652
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_8:arm/float/NCHW                                    {1,512,14,14}   0.131515        0.152164        262.648490
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_8:arm/float/NCHW                                    {1,512,14,14}   0.172808        0.239339        482.977186
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_9:arm/float/NCHW                                    {1,512,14,14}   0.100261        0.139923        298.286828
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_9:arm/float/NCHW                                    {1,512,14,14}   0.187229        0.249410        428.661308
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_10:arm/float/NCHW                                   {1,512,14,14}   0.095613        0.138695        304.580973
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_10:arm/float/NCHW                                   {1,512,14,14}   0.210138        0.274737        511.803464
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_11:arm/float/NCHW                                   {1,512,14,14}   0.089060        0.146312        430.252419
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_11:arm/float/NCHW                                   {1,512,14,14}   0.216164        0.308152        616.695792
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_12:arm/float/NCHW                                   {1,512,14,14}   0.065522        0.128159        319.802496
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_12:arm/float/NCHW                                   {1,512,7,7}     0.199970        0.350766        991.288010
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_13:arm/float/NCHW                                   {1,1024,7,7}    0.019780        0.076383        177.504131
depthwise_conv2d:arm/float/NCHW               batch_norm_1.tmp_3_13:arm/float/NCHW                                   {1,1024,7,7}    0.120542        0.246950        319.743111
conv2d:arm/float/NCHW                         batch_norm_0.tmp_3_14:arm/float/NCHW                                   {1,1024,7,7}    0.092912        0.328396        927.917227
pool2d:arm/float/NCHW                         batch_norm_1.tmp_3_14:arm/float/NCHW                                   {1,1024,1,1}    0.092912        0.287532        4342.388333
fc:arm/float/NCHW                             batch_norm_0.tmp_3_15:arm/float/NCHW                                   {1,1000}        0.000057        1.285440        -0.573233
softmax:arm/float/NCHW                        save_infer_model/scale_0.tmp_1_1:arm/float/NCHW                        {1,1000}        0.001000        0.002196        2.854896
[note]
1. `ave_grow_rate`: show the sequence value of tensor when std_dev & mean are same.
2. Enable write each output tensor to file: `export PADDLELITE_PRECISION_WRITE_TO_FILE=1` on ADB command line.
```

对每个 output tensor 除了有维度/设备/数据排布/精度信息外，还有 3 个数值来表示，用于快速核验：

1. 均值(`mean`)：该 tensor 所有元素的和值除以元素个数。反应整体的平均值情况；
2. 标准差(`std_deviation`)：该 tensor 距离均值的波动程度。一般来说，均值和标准差就能确定该 tensor 的正确性；
3. 序列值(`ave_grow_rate`)：该 tensor 从起始元素到最后一个元素的变化情况，反应整体的序列变化情况。当两组 tensor 均值和标准差一样，但是序列即出现的位次不同时，该值也不同。

> 序列值的算法
> 从第二个元素起始，通过对当前元素减去前一个数并对差值除以前一个数，并将所有值累加，并将累加值除以总的元素个数。
> 序列值的计算过程伪代码为：
> ```cpp
> // compute ave_grow_rate of tensor output
> for (size_t i = 1; i < output.length; ++i) {
>   ave_grow_rate += (output[i] - output[i - 1]) / (output[i - 1] + eps);
> }
> ave_grow_rate /= output.length;
> ```


此外，若要保存每个 OP 的每个数据输出到文件，可以在 ADB Shell 环境里执行前加入`export PADDLELITE_PRECISION_WRITE_TO_FILE=1`，就会将每层每个的输出写文件保存，若是多次执行则会将每次推理的结果按照`PaddleLite`为前缀加时间戳的命名方式，保存在不同的文件夹里，文件夹存储路径会在打印的信息中注明。


### Profiler 架构设计

- Op 层信息：`struct Instruction::SetProfileRuntimeOpInfo`方法中会调用`OpLite->GetOpRuntimeInfo(profile::OpCharacter*)`，由各个从`OpLite`派生出的子类 Op 重写如`./lite/operator/conv_op.h`中的`class ConvOpLite : public OpLite`重写了`GetOpRuntimeInfo`方法实现了对 Conv Op 信息获取，从而实现了在`Instruction::SetProfileRuntimeOpInfo`中获取每个 Op 的信息。
- Kernel 层信息：`class KernelBase::SetProfileRuntimeKernelInfo(profile::OpCharacter* ch)`方法为虚函数，实际执行会调用由`KernelBase`派生的最终子类，如`class ReluCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)>`，`class KernelLite`由`KernelBase`派生而来，实现多态机制下的 Kernel 信息获取，如具体的底层 Kernel 名。

通过在 Op 层将`OpLite*`成员以`void*`的形式放在`profile::OpCharacter`结构体中，并将此结构体传递给 Kernel 层，实现获取所有的 Op 层与 Kernel 层信息。
