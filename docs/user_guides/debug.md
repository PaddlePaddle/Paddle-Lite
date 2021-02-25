# 调试工具

## Profiler工具

Profiler 在 Lite 里分为性能 Profiler 和 精度 Profiler：

- 性能 Profiler ：用于逐层耗时统计，可以获取到模型逐层 ARM CPU / X86 CPU / OpenCL 上kernel 耗时信息。定位耗时潜在问题；
- 精度 Profiler ：用于逐层精度统计，可以获取到模型逐层每个 Op 的输出 tensor 精度信息。

### 开启方法

- 开启性能 Profiler: 修改根目录下`CMakeLists.txt`检索`lite_option`在`LITE_WITH_PROFILE`这一项，从`OFF`修改为`ON`；
- 开启精度 Profiler：修改根目录下`CMakeLists.txt`检索`lite_option`在`LITE_WITH_PRECISION_PROFILE`这一项，从`OFF`修改为`ON`。

修改后，针对移动端Android平台，按照下面方式编译：

```shell
# 根据指定编译参数编译
# 默认后接参数build_opencl
# 若是arm cpu模型不影响 Profiler 结果

./lite/tools/ci_build.sh \
  --arm_os=android \
  --arm_abi=[armv7|armv8] \
  --arm_lang=[gcc|clang] \
  build_opencl
```

其它平台，参照文档对各平台的编译方式进行编译。

### 精度 Profiler

对每个output tensor除了有维度/设备/数据排布/精度信息外，还3个数值来表示，用于快速核验：

1. 均值(`mean`)：表示该tensor所有元素的和值除以元素个数。反应整体的平均值情况；
2. 标准差(`std_deviation`)：表示该tensor距离均值的波动程度。一般来说，均值和标准差就能确定该 tensor 的正确性；
3. 序列值(`ave_grow_rate`)：表示该tensor从起始元素到最后一个元素的变化情况，反应整体的序列变化情况。该值用于当两组tensor均值和标准差一样，但是序列即出现的位次不同时，该值也不同。

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

在ADB shell环境，开启精度 Profiler 编译并运行模型如`caffe_mobilenetv1_opencl.nb`，会自动打印类似如下日志:

```shell
========================================= Detailed Precision Profiler Summary =========================================
operator:(kernel_info)                        output_tensor_name:(tensor_info)                                       dims            mean            std_deviation   ave_grow_rate*
io_copy:opencl/any/any                        data/target_trans_1:opencl/any/NCHW                                    {1,3,224,224}   1.000000        0.000000        0.000000
layout:opencl/any/ImageDefault                data/target_trans/layout_trans_1:opencl/any/ImageDefault               {1,3,224,224}   1.000000        0.000000        0.000000
conv2d:opencl/float16/ImageDefault            relu1.tmp_0_1:opencl/float16/ImageDefault                              {1,32,112,112}  0.282844        0.240534        11.435025
conv2d:opencl/float16/ImageDefault            relu2_1_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,32,112,112}  0.274217        0.283213        95.264143
conv2d:opencl/float16/ImageDefault            relu2_1_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,64,112,112}  0.088825        0.130261        149.436979
conv2d:opencl/float16/ImageDefault            relu2_2_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,64,56,56}    0.210590        0.295913        42.604240
conv2d:opencl/float16/ImageDefault            relu2_2_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,128,56,56}   0.157238        0.139949        29.510444
conv2d:opencl/float16/ImageDefault            relu3_1_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,128,56,56}   0.212104        0.225652        106.150392
conv2d:opencl/float16/ImageDefault            relu3_1_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,128,56,56}   0.064901        0.104246        88.708513
conv2d:opencl/float16/ImageDefault            relu3_2_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,128,28,28}   0.227582        0.250818        87.952434
conv2d:opencl/float16/ImageDefault            relu3_2_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,256,28,28}   0.112293        0.103471        36.863450
conv2d:opencl/float16/ImageDefault            relu4_1_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,256,28,28}   0.113568        0.165170        231.016689
conv2d:opencl/float16/ImageDefault            relu4_1_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,256,28,28}   0.066476        0.096654        76.182087
conv2d:opencl/float16/ImageDefault            relu4_2_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,256,14,14}   0.180241        0.217738        128.105476
conv2d:opencl/float16/ImageDefault            relu4_2_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,512,14,14}   0.091914        0.106345        71.584000
conv2d:opencl/float16/ImageDefault            relu5_1_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,512,14,14}   0.083399        0.130441        388.709359
conv2d:opencl/float16/ImageDefault            relu5_1_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,512,14,14}   0.070037        0.084776        157.475311
conv2d:opencl/float16/ImageDefault            relu5_2_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,512,14,14}   0.122580        0.156354        302.641378
conv2d:opencl/float16/ImageDefault            relu5_2_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,512,14,14}   0.062194        0.096692        263.157857
conv2d:opencl/float16/ImageDefault            relu5_3_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,512,14,14}   0.151563        0.204869        521.154652
conv2d:opencl/float16/ImageDefault            relu5_3_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,512,14,14}   0.069662        0.117923        474.364697
conv2d:opencl/float16/ImageDefault            relu5_4_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,512,14,14}   0.164965        0.244008        798.295050
conv2d:opencl/float16/ImageDefault            relu5_4_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,512,14,14}   0.078644        0.185467        822.652575
conv2d:opencl/float16/ImageDefault            relu5_5_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,512,14,14}   0.195893        0.337309        929.937706
conv2d:opencl/float16/ImageDefault            relu5_5_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,512,14,14}   0.073846        0.193728        787.249412
conv2d:opencl/float16/ImageDefault            relu5_6_dw.tmp_0_1:opencl/float16/ImageDefault                         {1,512,7,7}     0.229749        0.363817        1689.265557
conv2d:opencl/float16/ImageDefault            relu5_6_sep.tmp_0_1:opencl/float16/ImageDefault                        {1,1024,7,7}    0.011197        0.058381        118.107636
conv2d:opencl/float16/ImageDefault            relu6_dw.tmp_0_1:opencl/float16/ImageDefault                           {1,1024,7,7}    0.135260        0.162124        84.246188
conv2d:opencl/float16/ImageDefault            relu6_sep.tmp_0_1:opencl/float16/ImageDefault                          {1,1024,7,7}    0.142286        0.379486        1359.996046
pool2d:opencl/float16/ImageDefault            pool6.tmp_0_1:opencl/float16/ImageDefault                              {1,1024,1,1}    0.142280        0.314150        3670.321820
conv2d:opencl/float16/ImageDefault            fc7.tmp_1_1:opencl/float16/ImageDefault                                {1,1000,1,1}    0.001153        1.178804        -2.516469
layout:opencl/any/NCHW                        fc7.tmp_1/layout_trans_1:opencl/any/NCHW                               {1,1000,1,1}    0.001153        1.178804        -2.516469
io_copy:opencl/any/any                        fc7.tmp_1/target_trans_1:host/any/NCHW                                 {1,1000,1,1}    0.001153        1.178804        -2.516469
softmax:arm/float/NCHW                        prob_softmax.tmp_0_1:arm/float/NCHW                                    {1,1000,1,1}    0.001000        0.001865        1.862140
[note]
1. `ave_grow_rate`: show the sequence value of tensor when std_dev & mean are same.
2. Enable write each output tensor to file: `export PADDLELITE_PRECISION_WRITE_TO_FILE=1` on ADB command line.
```

此外，若要保存每个 OP 的输出到文件，可以在ADB Shell环境里执行前加入这句`export PADDLELITE_PRECISION_WRITE_TO_FILE=1`，就会将每层每个的输出写文件保存，若是多次执行则会将每次推理的结果按照`PaddleLite`为前缀加时间戳的命名方式，保存在不同的文件夹里，文件夹存储路径会在日志中注明。

### 性能 Profiler

在ADB shell环境，开启性能 Profiler 编译并运行模型，会自动打印类似如下日志，日志分为三部分：

1. Detailed Dispatch Profiler Summary：单次推理的逐 OP 底层 Kernel 层运行耗时，即在`KernelBase::Run()`的前后统计耗时。会排除第一次的计时（因为第一次不准确相当于wamrup），若多次执行也会打印多次；
2. Concise Create Profiler Summary：汇总统计的创建 Op 的耗时，即从`Instruction::Run()`开始到`KernelBase::Run()`执行前。会排除掉前 10 次推理；
3. Concise Dispatch Profiler Summary：汇总统计的运行 Op 的耗时，即在`KernelBase::Run()`的前后统计耗时，为 Lite 具体设备的底层 Kernel 层完整耗时，会排除掉前 10 次推理。

```shell
===== Detailed Dispatch Profiler Summary: N/A, Exclude 1 warm-ups =====
OperatorType         KerneAttr(Place)               KernelFuncName           Remark                     InDim           FilterDim       OutDim          Avg(ms) Min(ms) Max(ms) Last(ms) Avg(%)  GOPs    GOPS    clAvg(ms) clMin(ms) clMax(ms) clAvg(%)  GlobalWorkSize LocalWorkSize
io_copy              opencl/any/any                 HostToOpenCL             type0                      1x3x224x224     N/A             1x3x224x224     2.107   1.355   3.395   2.325   4.26%    0.000   0.00    2.094     1.345     3.376     2.86%    N/A          N/A
layout               opencl/any/ImageDefault        buffer_to_image2d        type0                      1x3x224x224     N/A             1x3x224x224     0.235   0.168   0.362   0.206   0.47%    0.000   0.00    0.281     0.207     0.330     0.38%    N/A          N/A
conv2d               opencl/float16/ImageDefault    conv2d_3x3_opt           3x3p1s2g1d1BiasRelu        1x3x224x224     32x3x3x3        1x32x112x112    1.925   1.213   2.604   2.016   3.89%    0.022   11.26   1.109     1.094     1.118     1.51%    8,23,112     1,23,7
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       3x3p1s1g32d1BiasRelu       1x32x112x112    32x1x3x3        1x32x112x112    1.448   0.819   1.911   1.531   2.93%    0.007   4.99    0.784     0.776     0.800     1.07%    8,56,112     4,56,1
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x32x112x112    64x32x1x1       1x64x112x112    1.361   1.147   1.637   1.633   2.75%    0.051   37.75   2.008     1.988     2.023     2.74%    16,28,112    1,28,7
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3         3x3p1s2g64d1BiasRelu       1x64x112x112    64x1x3x3        1x64x56x56      1.206   0.946   1.595   1.395   2.44%    0.004   3.00    0.708     0.654     0.774     0.97%    16,56,56     4,56,1
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x64x56x56      128x64x1x1      1x128x56x56     1.248   0.921   1.904   1.187   2.52%    0.051   41.16   1.973     1.942     1.993     2.69%    32,14,56     1,14,14
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       3x3p1s1g128d1BiasRelu      1x128x56x56     128x1x3x3       1x128x56x56     1.253   1.065   1.568   1.203   2.53%    0.007   5.76    0.752     0.744     0.766     1.03%    32,28,56     1,28,7
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x128x56x56     128x128x1x1     1x128x56x56     1.278   1.093   1.649   1.207   2.58%    0.103   80.38   3.852     3.846     3.867     5.26%    32,14,56     1,14,14
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3         3x3p1s2g128d1BiasRelu      1x128x56x56     128x1x3x3       1x128x28x28     1.311   1.145   1.426   1.277   2.65%    0.002   1.38    0.388     0.356     0.449     0.53%    32,28,28     1,28,7
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x128x28x28     256x128x1x1     1x256x28x28     1.224   1.068   1.369   1.336   2.47%    0.051   41.98   2.136     2.125     2.157     2.92%    64,7,28      1,7,28
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       3x3p1s1g256d1BiasRelu      1x256x28x28     256x1x3x3       1x256x28x28     1.458   1.108   1.995   1.452   2.95%    0.004   2.48    0.394     0.388     0.407     0.54%    64,14,28     1,14,14
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x256x28x28     256x256x1x1     1x256x28x28     1.382   1.016   1.806   1.334   2.80%    0.103   74.33   4.185     4.158     4.199     5.71%    64,7,28      1,7,28
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3         3x3p1s2g256d1BiasRelu      1x256x28x28     256x1x3x3       1x256x14x14     1.549   0.820   1.864   1.625   3.13%    0.001   0.58    0.237     0.208     0.271     0.32%    64,14,14     1,14,14
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x256x14x14     512x256x1x1     1x512x14x14     1.270   0.723   1.716   1.266   2.57%    0.051   40.47   2.806     2.457     3.316     3.83%    128,4,14     4,4,14
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     1.531   0.755   1.972   1.608   3.09%    0.002   1.18    0.235     0.227     0.243     0.32%    128,7,14     2,7,14
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     1.488   0.760   1.765   1.528   3.01%    0.103   69.07   5.528     4.851     6.542     7.54%    128,4,14     4,4,14
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     1.730   1.349   1.948   1.709   3.50%    0.002   1.04    0.232     0.220     0.239     0.32%    128,7,14     2,7,14
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     1.843   1.412   2.694   2.173   3.73%    0.103   55.76   5.693     4.851     6.533     7.77%    128,4,14     4,4,14
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     1.805   1.211   2.217   2.127   3.65%    0.002   1.00    0.234     0.227     0.247     0.32%    128,7,14     2,7,14
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     1.828   1.279   2.264   1.877   3.70%    0.103   56.22   5.693     4.839     6.539     7.77%    128,4,14     4,4,14
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     1.901   1.477   2.335   2.184   3.84%    0.002   0.95    0.233     0.222     0.250     0.32%    128,7,14     2,7,14
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     1.861   1.013   2.194   2.132   3.76%    0.103   55.22   6.031     4.846     6.542     8.23%    128,4,14     4,4,14
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     2.052   1.650   3.425   2.074   4.15%    0.002   0.88    0.234     0.228     0.250     0.32%    128,7,14     2,7,14
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     1.908   1.534   2.543   1.738   3.86%    0.103   53.87   5.698     4.852     6.540     7.77%    128,4,14     4,4,14
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3         3x3p1s2g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x7x7       1.917   1.619   2.331   2.023   3.87%    0.000   0.24    0.171     0.159     0.190     0.23%    128,7,7      4,7,7
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x512x7x7       1024x512x1x1    1x1024x7x7      1.760   1.558   2.086   1.687   3.56%    0.051   29.19   5.348     3.213     6.516     7.30%    256,2,7      16,2,7
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       3x3p1s1g1024d1BiasRelu     1x1024x7x7      1024x1x3x3      1x1024x7x7      1.755   1.430   2.109   1.764   3.55%    0.001   0.51    0.182     0.138     0.194     0.25%    256,4,7      8,4,7
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1BiasRelu        1x1024x7x7      1024x1024x1x1   1x1024x7x7      1.773   1.488   2.062   1.488   3.58%    0.103   57.97   10.346    6.379     12.993    14.12%    256,2,7      16,2,7
pool2d               opencl/float16/ImageDefault    pool_avg_global          globalavg                  1x1024x7x7      N/A             1x1024x1x1      0.201   0.166   0.222   0.205   0.41%    0.000   0.25    0.166     0.054     0.184     0.23%    N/A          N/A
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        1x1p0s1g1d1Biasunk         1x1024x1x1      1000x1024x1x1   1x1000x1x1      2.149   1.828   3.388   1.954   4.35%    0.002   0.95    2.072     0.767     2.234     2.83%    250,1,1      250,1,1
layout               opencl/any/NCHW                image2d_to_buffer        type0                      1x1000x1x1      N/A             1x1000x1x1      0.184   0.168   0.221   0.171   0.37%    0.000   0.00    0.030     0.011     0.039     0.04%    N/A          N/A
io_copy              opencl/any/any                 OpenCLToHost             type0                      1x1000x1x1      N/A             1x1000x1x1      1.460   0.744   2.007   0.744   2.95%    0.000   0.00    1.450     0.735     1.996     1.98%    N/A          N/A
softmax              arm/float/NCHW                 NotImpl                  axis1                      1x1000x1x1      N/A             1x1000x1x1      0.061   0.038   0.085   0.057   0.12%    0.000   0.10    0.000     0.000     0.000     0.00%    N/A          N/A

[I  2/23 18:45: 5.315 ...de/lite-doc-20210223/lite/core/program.h:195 ~RuntimeProgram]
Timing cycle = 11
===== Concise Create Profiler Summary: N/A, Exclude 10 warm-ups =====
OperatorType         KerneAttr(Place)               KernelFuncName           Avg(ms) Min(ms) Max(ms) Avg(%)  GOPs    CalledTimes clAvg(ms) clMin(ms) clMax(ms) clAvg(%)
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        0.240   0.240   0.240   43.48%    1.081   14          61.016    61.016    61.016    86.36%
conv2d               opencl/float16/ImageDefault    conv2d_3x3_opt           0.020   0.020   0.020   3.62%    0.022   1           1.094     1.094     1.094     1.55%
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3         0.073   0.073   0.073   13.22%    0.007   4           1.495     1.495     1.495     2.12%
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       0.160   0.160   0.160   28.99%    0.028   9           3.258     3.258     3.258     4.61%
io_copy              opencl/any/any                 HostToOpenCL             0.012   0.012   0.012   2.17%    0.000   1           1.366     1.366     1.366     1.93%
io_copy              opencl/any/any                 OpenCLToHost             0.006   0.006   0.006   1.09%    0.000   1           1.920     1.920     1.920     2.72%
layout               opencl/any/ImageDefault        buffer_to_image2d        0.010   0.010   0.010   1.81%    0.000   1           0.292     0.292     0.292     0.41%
layout               opencl/any/NCHW                image2d_to_buffer        0.007   0.007   0.007   1.27%    0.000   1           0.037     0.037     0.037     0.05%
pool2d               opencl/float16/ImageDefault    pool_avg_global          0.015   0.015   0.015   2.72%    0.000   1           0.179     0.179     0.179     0.25%
softmax              arm/float/NCHW                 NotImpl                  0.009   0.009   0.009   1.63%    0.000   1           0.000     0.000     0.000     0.00%

[I  2/23 18:45: 5.316 ...de/lite-doc-20210223/lite/core/program.h:196 ~RuntimeProgram]
Timing cycle = 11
===== Concise Dispatch Profiler Summary: N/A, Exclude 10 warm-ups =====
OperatorType         KerneAttr(Place)               KernelFuncName           Avg(ms) Min(ms) Max(ms) Avg(%)  GOPs    CalledTimes clAvg(ms) clMin(ms) clMax(ms) clAvg(%)
conv2d               opencl/float16/ImageDefault    conv2d_1x1_simple        22.540  22.540  22.540  44.87%    1.081   14          69.537    69.537    69.537    88.19%
conv2d               opencl/float16/ImageDefault    conv2d_3x3_opt           2.016   2.016   2.016   4.01%    0.022   1           1.112     1.112     1.112     1.41%
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3         6.320   6.320   6.320   12.58%    0.007   4           1.517     1.517     1.517     1.92%
conv2d               opencl/float16/ImageDefault    depth_conv2d_3x3s1       15.652  15.652  15.652  31.16%    0.028   9           3.293     3.293     3.293     4.18%
io_copy              opencl/any/any                 HostToOpenCL             2.325   2.325   2.325   4.63%    0.000   1           2.311     2.311     2.311     2.93%
io_copy              opencl/any/any                 OpenCLToHost             0.744   0.744   0.744   1.48%    0.000   1           0.735     0.735     0.735     0.93%
layout               opencl/any/ImageDefault        buffer_to_image2d        0.206   0.206   0.206   0.41%    0.000   1           0.284     0.284     0.284     0.36%
layout               opencl/any/NCHW                image2d_to_buffer        0.171   0.171   0.171   0.34%    0.000   1           0.011     0.011     0.011     0.01%
pool2d               opencl/float16/ImageDefault    pool_avg_global          0.205   0.205   0.205   0.41%    0.000   1           0.054     0.054     0.054     0.07%
softmax              arm/float/NCHW                 NotImpl                  0.057   0.057   0.057   0.11%    0.000   1           0.000     0.000     0.000     0.00%

[I  2/23 18:45: 5.338 ...210223/lite/backends/opencl/cl_context.h:42 ~CLContext] release cl::Program, cl::Kernel finished.
```

上面是 Android 端 OpenCL 的性能 Profiler 结果，根据 KernelFuncName 可以很轻松定位到当前执行的具体哪个 CL Kernel。根据耗时百分占比，可以进一步分析潜在性能问题。下面也给出 Arm CPU 在`caffe_mobilenetv1_arm.nb`模型的 Profiler 日志，作为参考：

```shell
===== Detailed Dispatch Profiler Summary: N/A, Exclude 1 warm-ups =====
OperatorType         KerneAttr(Place)               KernelFuncName           Remark                     InDim           FilterDim       OutDim          Avg(ms) Min(ms) Max(ms) Last(ms) Avg(%)  GOPs    GOPS    clAvg(ms) clMin(ms) clMax(ms) clAvg(%)  GlobalWorkSize LocalWorkSize
conv2d               arm/float/NCHW                 conv_3x3s2_direct_fp32   3x3p1s2g1d1BiasRelu        1x3x224x224     32x3x3x3        1x32x112x112    2.220   1.438   3.317   1.646   2.70%    0.022   9.77    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g32d1BiasRelu       1x32x112x112    32x1x3x3        1x32x112x112    1.128   0.710   1.512   1.036   1.37%    0.007   6.40    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x32x112x112    64x32x1x1       1x64x112x112    4.458   2.606   6.471   4.047   5.41%    0.051   11.52   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s2g64d1BiasRelu       1x64x112x112    64x1x3x3        1x64x56x56      1.127   0.652   1.663   0.744   1.37%    0.004   3.21    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x64x56x56      128x64x1x1      1x128x56x56     3.929   2.319   6.038   2.661   4.77%    0.051   13.08   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g128d1BiasRelu      1x128x56x56     128x1x3x3       1x128x56x56     1.210   0.721   1.871   0.844   1.47%    0.007   5.97    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x128x56x56     128x128x1x1     1x128x56x56     7.373   4.369   11.180  5.430   8.95%    0.103   13.94   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s2g128d1BiasRelu      1x128x56x56     128x1x3x3       1x128x28x28     0.614   0.338   0.922   0.421   0.75%    0.002   2.94    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x128x28x28     256x128x1x1     1x256x28x28     3.554   2.006   5.190   2.520   4.32%    0.051   14.46   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g256d1BiasRelu      1x256x28x28     256x1x3x3       1x256x28x28     0.584   0.341   0.887   0.421   0.71%    0.004   6.19    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x256x28x28     256x256x1x1     1x256x28x28     6.736   3.926   10.207  4.568   8.18%    0.103   15.26   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s2g256d1BiasRelu      1x256x28x28     256x1x3x3       1x256x14x14     0.370   0.202   0.549   0.232   0.45%    0.001   2.44    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x256x14x14     512x256x1x1     1x512x14x14     3.498   1.861   4.929   2.153   4.25%    0.051   14.69   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.391   0.208   0.545   0.239   0.47%    0.002   4.63    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     7.166   3.840   10.262  4.421   8.70%    0.103   14.34   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.389   0.198   0.593   0.238   0.47%    0.002   4.64    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     6.428   3.694   10.044  4.312   7.81%    0.103   15.99   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.337   0.199   0.530   0.222   0.41%    0.002   5.36    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     6.007   3.697   9.141   4.108   7.29%    0.103   17.11   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.425   0.204   1.438   0.222   0.52%    0.002   4.25    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     5.909   3.692   9.017   4.132   7.18%    0.103   17.39   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x14x14     0.325   0.199   0.531   0.220   0.40%    0.002   5.55    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x14x14     512x512x1x1     1x512x14x14     6.037   3.664   9.945   4.101   7.33%    0.103   17.02   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s2g512d1BiasRelu      1x512x14x14     512x1x3x3       1x512x7x7       0.196   0.114   0.296   0.125   0.24%    0.000   2.31    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x512x7x7       1024x512x1x1    1x1024x7x7      3.275   1.918   5.060   2.151   3.98%    0.051   15.69   0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  3x3p1s1g1024d1BiasRelu     1x1024x7x7      1024x1x3x3      1x1024x7x7      0.280   0.167   0.425   0.167   0.34%    0.001   3.23    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1BiasRelu        1x1024x7x7      1024x1024x1x1   1x1024x7x7      7.273   4.147   12.180  4.264   8.83%    0.103   14.13   0.000     0.000     0.000     0.00%    N/A          N/A
pool2d               arm/float/NCHW                 NotImpl                  globalavg                  1x1024x7x7      N/A             1x1024x1x1      0.028   0.017   0.043   0.017   0.03%    0.000   1.78    0.000     0.000     0.000     0.00%    N/A          N/A
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      1x1p0s1g1d1Biasunk         1x1024x1x1      1000x1024x1x1   1x1000x1x1      1.060   0.666   1.296   0.858   1.29%    0.002   1.93    0.000     0.000     0.000     0.00%    N/A          N/A
softmax              arm/float/NCHW                 NotImpl                  axis1                      1x1000x1x1      N/A             1x1000x1x1      0.016   0.011   0.023   0.011   0.02%    0.000   0.38    0.000     0.000     0.000     0.00%    N/A          N/A

[I  2/23 18:48:24.825 ...de/lite-doc-20210223/lite/core/program.h:195 ~RuntimeProgram]
Timing cycle = 11
===== Concise Create Profiler Summary: N/A, Exclude 10 warm-ups =====
OperatorType         KerneAttr(Place)               KernelFuncName           Avg(ms) Min(ms) Max(ms) Avg(%)  GOPs    CalledTimes clAvg(ms) clMin(ms) clMax(ms) clAvg(%)
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      0.085   0.085   0.085   52.80%    1.081   14          0.000     0.000     0.000     0.00%
conv2d               arm/float/NCHW                 conv_3x3s2_direct_fp32   0.006   0.006   0.006   3.73%    0.022   1           0.000     0.000     0.000     0.00%
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  0.063   0.063   0.063   39.13%    0.035   13          0.000     0.000     0.000     0.00%
pool2d               arm/float/NCHW                 NotImpl                  0.004   0.004   0.004   2.48%    0.000   1           0.000     0.000     0.000     0.00%
softmax              arm/float/NCHW                 NotImpl                  0.003   0.003   0.003   1.86%    0.000   1           0.000     0.000     0.000     0.00%

[I  2/23 18:48:24.826 ...de/lite-doc-20210223/lite/core/program.h:196 ~RuntimeProgram]
Timing cycle = 11
===== Concise Dispatch Profiler Summary: N/A, Exclude 10 warm-ups =====
OperatorType         KerneAttr(Place)               KernelFuncName           Avg(ms) Min(ms) Max(ms) Avg(%)  GOPs    CalledTimes clAvg(ms) clMin(ms) clMax(ms) clAvg(%)
conv2d               arm/float/NCHW                 conv1x1s1_gemm_fp32      49.726  49.726  49.726  87.96%    1.081   14          0.000     0.000     0.000     0.00%
conv2d               arm/float/NCHW                 conv_3x3s2_direct_fp32   1.646   1.646   1.646   2.91%    0.022   1           0.000     0.000     0.000     0.00%
conv2d               arm/float/NCHW                 conv_depthwise_3x3_fp32  5.131   5.131   5.131   9.08%    0.035   13          0.000     0.000     0.000     0.00%
pool2d               arm/float/NCHW                 NotImpl                  0.017   0.017   0.017   0.03%    0.000   1           0.000     0.000     0.000     0.00%
softmax              arm/float/NCHW                 NotImpl                  0.011   0.011   0.011   0.02%    0.000   1           0.000     0.000     0.000     0.00%

[I  2/23 18:48:24.833 ...10223/lite/backends/opencl/cl_runtime.cc:33 ~CLRuntime] is_cl_runtime_initialized_:1
```

### Profiler 架构设计

- Op 层信息：`struct Instruction::SetProfileRuntimeOpInfo`方法中会调用`OpLite->GetOpRuntimeInfo(profile::OpCharacter*)`，由各个从`OpLite`派生出的子类Op重写如`./lite/operator/conv_op.h`中的`class ConvOpLite : public OpLite`重写了`GetOpRuntimeInfo`方法实现了对 Conv Op 信息获取；
- Kernel 层信息：`class KernelBase::SetProfileRuntimeKernelInfo(profile::OpCharacter* ch)`方法为虚函数，实际执行会调用由`KernelBase`派生的最终子类，如`class ReluCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)>`，`class KernelLite`由`KernelBase`派生而来，实现多态机制下的 Kernel 信息获取，如具体的底层 Kernel名。

通过在 Op 层将`OpLite*`成员以`void*`的形式挂载到`profile::OpCharacter`，并传递给 Kernel 层，实现获取所有的 Op 层与 Kernel 层信息。

## Debug工具

**Lite Model Debug Tool** 是用来检查Paddle-Lite框架与Paddle-Fluid框架运行时tensor(包括variable与weight)之间diff信息的基础工具。

### 编译方法:

1. 参照 [编译环境准备](../source_compile/compile_env) 进行环境配置和编译。
2. 在生成的`build`目录下，执行`make lite_model_debug_tool`，`lite_model_debug_tool`产出在编译目录的`lite/tools/debug`目录下。

### 工作流程:

1. 运行 `/bin/bash check_model.sh --model_dir=<your_model_path> --build_root_dir=<your_cmake_root_dir> debug_cpp_stage` 获得模型在Paddle-Lite框架下的运行拓扑信息、varibles信息和weights信息。运行后拓扑信息将会存储在默认名为 `topo_file.txt` 的文件中，variables和weights信息将会存储在默认名为 `tensor_cpp.txt` 的文件中。
2. 运行 `/bin/bash check_model.sh --model_dir=<your_model_path> --build_root_dir=<your_cmake_root_dir> debug_py_stage`执行fluid框架预测以获取相同模型在fluid框架下的variable与weight信息(注意：我们使用fluid的python api运行fluid模型，因此您在运行此步之前应确保已正确安装fluid的python api)。然后debug tool将会自动比较Paddle-Lite框架输出的信息和Paddle-Fluid框架输出的信息来检查是否存在运行时diff。 执行Paddle-Fluid框架，输出的信息将会存储在默认名为 `tensor_py.txt` 的文件中，相应的diff信息将会存储在默认名为 `diff.txt`的文件中(默认情况下，只会输出执行拓扑序中第一个有diff的variable相关的信息)。

### 注意事项:

1. 输出的结果是在**执行完一次预测后**输出的相应变量/权重的最终值，因此如果您在预测过程进行过诸如变量复用/子图融合等优化方法，则相应的输出可能会出现偏差。
2. 默认情况下debug tools将以全1作为输入进行比对。
3. 默认情况下，为了保证与Paddle-Fluid框架的结果可比对，debug tool将会禁用掉所有的Paddle-Lite的优化策略。
4. Paddle-Lite框架的执行环境由与您的编译选项有关，比如您开启了LITE_WITH_ARM编译选项，那debug tool的`debug_cpp_stage`也需要在ARM平台下运行。

### Diff信息输出：

如果debug tool检测到diff信息，那么在`diff.txt`中将会输出类似以下结构信息

```c++
>>>>>>>>>>>>>>>>>>DIFF VARIABLE: dropout_0.tmp_0<<<<<<<<<<<<<<<<<<<
dropout	(X:pool2d_7.tmp_0)	(Mask:dropout_0.tmp_1 Out:dropout_0.tmp_0)
--------------- Tensor File info ---------------
pool2d_7.tmp_0	{1,1536,1,1}	0.749892 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0150336 0.621641 0.147099 0.636727 0.0 0.0 0.00410917 0.784708 0.0 0.0704846 0.233599 0.840123 0.239201 0.112878 0.0 0.155352 0.306906 0.0 0.0 0.860938 0.221037 0.787316 0.256585 ...
dropout_0.tmp_0	{1,1536,1,1}	0.749892 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0150336 0.621641 0.147099 0.636727 0.0 0.0 0.00410917 0.784708 0.0 0.0704846 0.233599 0.840123 0.239201 0.112878 0.0 0.155352 0.306906 0.0 0.0 0.860938 0.221037 0.787316 0.256585 ...
--------------- Fluid Tensor info ---------------
pool2d_7.tmp_0	{1,1536,1,1}	0.7498912 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.015033395 0.6216395 0.14709876 0.63672537 0.0 0.0 0.0041093696 0.7847073 0.0 0.07048465 0.23359808 0.8401219 0.23919891 0.1128789 0.0 0.1553514 0.3069055 0.0 0.0 0.8609365 0.22103554 ...
dropout_0.tmp_0	{1,1536,1,1}	0.599913 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.012026716 0.4973116 0.117679015 0.5093803 0.0 0.0 0.0032874958 0.62776583 0.0 0.056387722 0.18687847 0.67209756 0.19135913 0.090303116 0.0 0.12428112 0.2455244 0.0 0.0 0.68874925 ...
```

其中第二行为op相关信息，标明了执行哪个op出现了diff及其对应的输入输出变量名。Tensor File info为Paddle-Lite框架的输出信息，而Fluid Tensor info为Paddle-Fluid框架的相应输出信息。
示例中的`dropout_0.tmp_1`没有相应的tensor信息是因为工具检测到其在预测的后序流程中未被使用，因此不会对预测结果造成影响，从而将其自动屏蔽掉以保证输出尽量简洁。

### 其他选项：

| Option                      | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| --input_file                | 输入文件名，不同field以逗号分隔，相同field内以空格分隔, 只有文件中的第一行输入信息会被使用. 如果您不指定input_file，那么所有输入将会被置为1。注意：`debug_py_stage`目前不支持多field输入。 |
| --cpp_topo_file             | 存储运行时拓扑信息，由`debug_cpp_stage`写入并且由`debug_py_stage`读取使用。 默认为`topo_file.txt` 。 |
| --cpp_tensor_file           | 存储`debug_cpp_stage` 在运行拓扑序下的输出信息，默认为 `tensor_cpp.txt` 。 |
| --tensor_names              | 如果此选项不为空，那么只输出由此选项中指定名字的variable/weight信息，名字间用逗号分隔。 |
| --tensor_output_length      | 输出数据的长度，默认为全部输出。                             |
| --py_threshold              | 判断diff发生的阈值，默认为 `1e-5` 。                         |
| --py_tensor_file            | 存储`debug_py_stage` 在运行拓扑序下的输出信息，默认为`tensor_py.txt`. |
| --py_output_file            | diff信息的存储文件，默认为`diff.txt`。                       |
| --py_only_output_first_diff | 是否只输出运行时拓扑序中第一个有diff的var/op信息，默认为true |

您可以参考 `check_model.sh` 脚本中的代码以获得更多细节.
