---
name: 预测（Inference Issue）
about: 您可以提问预测中报错、应用等问题。 You could use this template for reporting an inference issue.

---

为使您的问题得到快速解决，在建立 Issue 前，请您先通过如下方式搜索是否有相似问题:[历史 issue](https://github.com/PaddlePaddle/Paddle-Lite/issues), [FAQ 文档](https://www.paddlepaddle.org.cn/lite/develop/quick_start/faq.html), [官方文档](https://www.paddlepaddle.org.cn/lite/develop/guide/introduction.html)

如果您没有查询到相似问题，为快速解决您的提问，建立 issue 时请提供如下细节信息：
- 标题：简洁、精准描述您的问题，例如“最新预测库的 API 文档在哪儿 ”
- 版本、预测库信息：
    1）Paddle Lite 版本：请提供您的 Paddle Lite 版本号（如v2.10）或 CommitID
    2）Host 环境：请描述 Host 系统类型、OS 版本，如 Mac OS 10.14、Ubuntu 18.04
    3）运行设备环境：请描述运行设备信息，如小米 9、iPhone13
    4）预测后端信息：请描述使用的预测后端信息，如 CPU/GPU/NPU/others 加速器  
- 预测信息
    1）预测 API：C++/JAVA/Python API
    2）预测选项信息：armv7/armv8、单线程/多线程 等
    3）预测库来源：官网下载/源码编译，如果是源码编译，辛苦提供源码编译命令，如 clang/gcc、openmp 等
- 复现信息：如为报错，请提供剥离业务环境的可复现预测 demo，用于问题复现
- 问题描述：请详细描述您的问题，同步贴出报错信息、日志/代码关键片段
