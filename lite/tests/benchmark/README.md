# 运行方式
```shell
-- cd Paddle-Lite/lite/tests/benchmark
-- ./build_benchmark_ops.sh #把build目录下的所有单测可执行文件push到手机上
在build_benchmark_ops.sh中运行python get_latency_lookup_table.py --ops_path ops.txt  --latency_lookup_table_path latency_lookup_table.txt
其中ops.txt是输入的网络模型文件， latency_lookup_table.txt是执行lite单测后输出的网络op耗时信息文件。
```
# 输入ops.txt格式说明
-- op_name  [dim0 dim1 dim2 dim3]   (op_param0, op_param1, ...， dtype=xxx)
   ops.txt每一行有三个字段，第一个字段是op_name, 第二个字段是输入Tensor的input_dims,
   第三个字段用()括起来，描述该op的parameter.
   # 注意： 每一个字段之间是以tab来分割的，parameter内的子字段是以逗号来分割的，
   # 描述tensor维度的[]内的数据之间以空格来分割，不能加逗号和tab.
   op_name现支持取值为conv/activation/batchnorm/pooling/fc;
   input_dims描述的是输入tensor格式，支持NCHW 4D等Tensor格式;
   op_param0,op_param1等字段描述该op的param属性，比如conv op包含ch_out/stride/group/kernel/pad/dilation/flag_bias/flag_act等属性;
   dtype描述该层op使用的数据类型，支持的合法输入为float/int8_float/int8_int8, 现在conv支持三种数据类型，其他op只支持float一种数据类型.
   
   # conv op格式
   conv  [1 96 112 112] (ch_out=48, stride=1, group=1, kernel=1x1, pad=0, dilation=1, flag_bias=0, flag_act=0, dtype=float)
   ch_out表示输出channel值， kernel表示卷积核size, 支持的合法取值为1x1/3x3/5x5等, pad表示边界padding的取值， flag_bias表示是否有bias, flag_act表示是否融合激活函数，支持的合法取值为0/1/2/4.
   
   # activitation op格式
   activation  [1 8 64 64] (act_type=relu)
   act_type表示激活函数类型，合法取值为relu/relu6/leaky_relu/tanh/swish/exp/abs/hard_swish/reciprocal/threshold_relu.

   # batchnorm op格式
   batchnorm   [1 8 64 64] (epsilon=1e-4f, momentum=0.9f)
   epsilon表示batchnorm的epsilon参数取值， 默认值为1e-4f;
   momentum表示batchnorm的momentum参数取值， 默认值为0.9f.

   # pooling op格式
   pooling  [1 8 64 64] (stride=2, pad=0, kernel=2x2, ceil_mode=0, flag_global=0, exclusive=1, pooling_type=max)
   stride表示pooling操作的跨度，默认值取2;pad表示边界padding的取值，默认值取0;
   kernel表示pooling卷积核size, 常见取值为2x2(默认值)；
   ceil_mode表示pooling是否进行ceil操作，=0表示false(默认值)，否则表示为true;
   flag_global表示pooling是否在WxH维度进行全局操作，=0表示false(默认值)，否则表示为true;
   exclusive表示pooling操作时的exclusive取值，=1表示true(默认值)，否则表示为false;
   pooling_type表示pooling类型，合法取值为max(默认值)/avg.

   # fc op格式
   fc [1 64]   (flag_bias=1, param_dim=64x1000)
   flag_bias表示fc op是否有bias，=1(默认值)表示为true, 否则为false;
   param_dim表示fc op `k x n`的操作维度信息，其中k应与input_dims=[m k]中的k取值保持一致.
   
# 输出latency_lookup_table.txt格式说明
dev_info           core_num thread_num	power_mode	core0 arch	core1 arch	core2 arch	core3 arch	core4 arch	core5 arch	core6 arch	core7 arch
Hisilicon Kirin980    8       1         	0         ARM_A55  	ARM_A55  	ARM_A55  	ARM_A55  	ARM_A76  	ARM_A76  	ARM_A76  	ARM_A76

op_name   	input_dims	   output_dims	   param_info     min_latency(ms)	  max_latency(ms)	 avg_latency(ms)
conv      	[1 96 112 112]	[1 48 114 114]	(ch_out=48, stride=1, pad=0, kernel=1x1, group=1, dilation=1, flag_bias=0, flag_act=0, dtype=float) 	3.469     	4.111     	3.52088   
fc        	[1 64]   	[64 1000] 	(param_dim=64x1000, flag_bias=1, dtype=float)  0.135     	0.176     	0.13779   
batchnorm 	[1 8 64 64]	[1 8 64 64]	(epsilon=1e-4f, momentum=0.9f, dtype=float)    0.014     	0.178     	0.01679   
pooling   	[1 8 64 64]	[1 8 32 32]	(stride=2, pad=0, kernel=2x2, ceil_mode=0, flag_global=0, exclusive=0, pooling_type=avg, dtype=float) 	0.009     	0.011     	0.00983   
activation	[1 8 64 64]	[1 8 64 64]	(act_type=relu, dtype=float)                   0.01      	0.036     	0.01103

-- 第一栏为header信息栏， 包含`dev_info` `arm_v7/v8` `core_num` `thread_num` `power_mode` `core0 arch` ... `core7 arch`字段：
   `dev_info`表示手机hardware厂家型号信息， `arm_v7/v8`表示armv7还是armv8架构, `core_num`表示cpu核心数， `thread_num`表示设置的运行多线程数，
   `power_mode`表示cpu绑核方式，
   `core0 arch`...`core7 arch`表示arm cpu架构信息
   第二栏为op信息栏， 包含`op_name` `input_dims` `output_dims` `param_info` `min_latency` `max_latency` `avg_latency`字段：
   其中`output_dims`为该层op根据`input_dims`和`param_info`计算得到的输出tensor维度信息;
   `min_latency(ms)` `max_latency(ms)` `avg_latency(ms)`为该层op运行得到的min/max/avg耗时信息.
