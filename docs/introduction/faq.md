# FAQ 常见问题

问题或建议可以发Issue，为加快问题解决效率，可先检索是否有类似问题，我们也会及时解答！
欢迎加入Paddle-Lite百度官方QQ群：696965088

1. 在Host端采用交叉编译方式编译PaddleLite，将编译后的libpaddle_light_api_shared.so和可执行程序放到板卡上运行，出现了如下图所示的错误，怎么解决？ 
![host_target_compiling_env_miss_matched](https://user-images.githubusercontent.com/9973393/75761527-31b8b700-5d74-11ea-8a9a-0bc0253ee003.png)
- 原因是Host端的交叉编译环境与Target端板卡的运行环境不一致，导致libpaddle_light_api_shared.so链接的GLIBC库高于板卡环境的GLIBC库。目前有四种解决办法（为了保证编译环境与官方一致，推荐第一种方式）：1）在Host端，参考[编译环境准备](../source_compile/compile_env)和[Linux源码编译](../source_compile/compile_linux)中的Docker方式重新编译libpaddle_light_api_shared.so；2）在Host端，使用与Target端版本一致的ARM GCC和GLIBC库重新编译libpaddle_light_api_shared.so；3）在Target端板卡上，参考[编译环境准备](../source_compile/compile_env)和[Linux源码编译](../source_compile/compile_linux)中的ARM Linux本地编译方式重新编译libpaddle_light_api_shared.so；4）在Target端板卡上，将GLIBC库升级到和Host端一致的版本，即GLIBC2.27。

2.Paddle Lite支持英伟达的Jetson硬件吗？
答：对于英伟达的Jetson硬件，用户应该使用paddle inference。使用Paddle Inference有两个优势：1. Paddle Inference是PaddlePaddle的原生推理库，与NV的Jeston硬件做了深度的适配，在该硬件上支持的所有Paddle原生的模型 2. 使用Paddle inference，可以利用TRT加速推理。 因为Jetson硬件本身是arm cpu+ NV gpu的异构硬件，如果只是使用Jetson的cpu做推理的话，是可以使用Paddle Lite的，但是显然，这样会浪费该硬件的GPU能力。
Paddle Inference文档：
https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id1
https://paddle-inference.readthedocs.io/en/latest/#

3.使用OPT工具转换失败的一般步骤
答：首先确认必须要用相同版本的opt和预测库。如果报错有某个op不支持，那么需要opt新增对该算子的支持。 使用./opt --print_all_ops=true, 可以查看当前Paddle Lite支持的所有op。另外，你也可以尝试编译develop版本的代码，有可能对未支持的op做了支持，但是尚未合入到稳定版本。

4.如果出现找不到 '__model__'文件、version不匹配、加载模型时segmentation_fault等问题，怎么办？
答：这时候，请仔细检查模型的输入路径是否正确。如果输入的路径不正确，容易出现以上问题。比如需要注意，combined model和uncombined model的参数输入方式是不同。前者的输入需要通过两个参数--model_file，--param_file来分别指定模型和参数文件，后者只需要一个参数--model_file来指定模型的目录。
