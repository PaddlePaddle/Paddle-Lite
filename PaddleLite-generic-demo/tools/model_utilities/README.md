# 使用说明
## split_model.py
### 目的
将 Paddle 推理模型裁剪到某个中间输出，得到一个新的 Paddle 推理模型。
### 步骤
- 安装 paddlepadle 2.2.2（或最新版本） whl 包；
  ```
  $ pip install paddlepaddle==2.2.2
  ```
- 将待处理模型目录的 model.pdmodel 或 __model__ 拖到 Netron https://netron.app/ 进行可视化；
- 找到需要截断的 op 并记录该 op 的输出 tensor 的名字，例如示例模型中的第一个 conv2d 的输出 tensor 名字为 conv2d_0.tmp_0 ；
- 在 split_model.py 中找到 SRC_MODEL_DIR = "./simple_model" ，把 ./simple_model 改成待处理的模型目录绝对路径；
- 在 split_model.py 中找到 fetch_targets = [program.block(0).var("conv2d_0.tmp_0")] ，把 conv2d_0.tmp_0 改成你之前记录的输出 tensor 的名字；
- 运行 python ./split_model.py 后即可生成裁剪后的模型 output_model ；
- 将 output_model 目录中的 model.pdmodel 或 __model__ 拖到 Netron https://netron.app/ 确认模型是否裁剪成功。
