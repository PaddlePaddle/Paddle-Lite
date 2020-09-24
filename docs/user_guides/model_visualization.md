# 模型可视化方法

Paddle Lite框架中主要使用到的模型结构有2种：(1) 为[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)深度学习框架产出的模型格式; (2) 使用[Lite模型优化工具opt](model_optimize_tool)优化后的模型格式。因此本章节包含2部分内容：

1. [Paddle推理模型可视化](model_visualization.html#paddle)
2. [Lite优化模型可视化](model_visualization.html#lite)

## Paddle推理模式可视化

Paddle用于推理的模型是通过[save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/io_cn/save_inference_model_cn.html#save-inference-model)这个API保存下来的，存储格式有两种，由save_inference_model接口中的 `model_filename` 和 `params_filename` 变量控制：

- **non-combined形式**：参数保存到独立的文件，如设置 `model_filename` 为 `None` , `params_filename` 为 `None`

  ```bash
  $ ls -l recognize_digits_model_non-combined/
  total 192K
  -rw-r--r-- 1 root root  28K Sep 24 09:39 __model__          # 模型文件
  -rw-r--r-- 1 root root  104 Sep 24 09:39 conv2d_0.b_0       # 独立权重文件
  -rw-r--r-- 1 root root 2.0K Sep 24 09:39 conv2d_0.w_0       # 独立权重文件
  -rw-r--r-- 1 root root  224 Sep 24 09:39 conv2d_1.b_0       # ...
  -rw-r--r-- 1 root root  98K Sep 24 09:39 conv2d_1.w_0
  -rw-r--r-- 1 root root   64 Sep 24 09:39 fc_0.b_0
  -rw-r--r-- 1 root root  32K Sep 24 09:39 fc_0.w_0
  ```

- **combined形式**：参数保存到同一个文件，如设置 `model_filename` 为 `model` , `params_filename` 为 `params`

  ```bash
  $ ls -l recognize_digits_model_combined/
  total 160K
  -rw-r--r-- 1 root root  28K Sep 24 09:42 model         # 模型文件
  -rw-r--r-- 1 root root 132K Sep 24 09:42 params        # 权重文件
  ```

通过以上方式保存下来的模型文件都可以通过[Netron](https://lutzroeder.github.io/netron/)工具来打开查看模型的网络结构。

**注意：**[Netron](https://github.com/lutzroeder/netron)当前要求PaddlePaddle的保存模型文件名必须为`__model__`，否则无法识别。如果是通过第二种方式保存下来的combined形式的模型文件，需要将文件重命名为`__model__`。



## Lite优化模型可视化

Paddle Lite在执行模型推理之前需要使用[模型优化工具opt](model_optimize_tool)来对模型进行优化，优化后的模型结构同样可以使用[Netron](https://lutzroeder.github.io/netron/)工具进行查看，但是必须保存为`protobuf`格式，而不是`naive_buffer`格式。

**注意**: 为了减少第三方库的依赖、提高Lite预测框架的通用性，在移动端使用Lite API您需要准备Naive Buffer存储格式的模型。

这里以[paddle_lite_opt](opt/opt_python)工具为例：

- 当模型输入为`non-combined`格式的Paddle模型时，需要通过`--model_dir`来指定模型文件夹

  ```bash
  $ paddle_lite_opt \
        --model_dir=./recognize_digits_model_non-combined/ \
        --valid_targets=arm \
        --optimize_out_type=protobuf \ # 注意：这里必须输出为protobuf格式
        --optimize_out=model_opt_dir_non-combined
  ```

  优化后的模型文件会存储在由`--optimize_out`指定的输出文件夹下，格式如下

  ```bash
  $ ls -l model_opt_dir_non-combined/
  total 152K
  -rw-r--r-- 1 root root  17K Sep 24 09:51 model     # 优化后的模型文件
  -rw-r--r-- 1 root root 132K Sep 24 09:51 params    # 优化后的权重文件
  ```

- 当模式输入为`combined`格式的Paddle模型时，需要同时输入`--model_file`和`--param_file`来分别指定Paddle模型的模型文件和权重文件

  ```bash
  $ paddle_lite_opt \
        --model_file=./recognize_digits_model_combined/model \
        --param_file=./recognize_digits_model_combined/params \
        --valid_targets=arm \
        --optimize_out_type=protobuf \ # 注意：这里必须输出为protobuf格式
        --optimize_out=model_opt_dir_combined
  ```
  优化后的模型文件同样存储在由`--optimize_out`指定的输出文件夹下，格式相同

  ```bash
  ls -l model_opt_dir_combined/
  total 152K
  -rw-r--r-- 1 root root  17K Sep 24 09:56 model     # 优化后的模型文件
  -rw-r--r-- 1 root root 132K Sep 24 09:56 params    # 优化后的权重文件
  ```


将通过以上步骤输出的优化后的模型文件`model`重命名为`__model__`，然后用[Netron](https://lutzroeder.github.io/netron/)工具打开即可查看优化后的模型结构。将优化前后的模型进行对比，即可发现优化后的模型比优化前的模型更轻量级，在推理任务中耗费资源更少且执行速度也更快。

<p align="center"><img width="600" src="https://paddlelite-data.bj.bcebos.com/doc_images/model_visualization/model_visualization.png"/></p>
