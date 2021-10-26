
裁剪预测库
============

Paddle Lite 支持根据模型裁剪预测库功能。Paddle Lite 的一般编译会将所有已注册的算子打包到预测库中，造成库文件体积膨胀；裁剪预测库能针对具体的模型，只打包优化后该模型需要的算子，有效降低预测库文件大小。

效果展示(Android 动态预测库体积)
--------------------------------

.. list-table::
   :header-rows: 1

   * - 测试模型
     - 裁剪开关
     - **libpaddle_light_api_shared.so**
     - 转化后模型中的 OP
   * - mobilenetv1（armv8）
     - 裁剪前
     - 1.5 MB
     - conv2d,depthwise_conv2d,fc,pool2d,softmax
   * - mobilenetv1（armv8）
     - 裁剪后
     - 859 KB
     - conv2d,depthwise_conv2d,fc,pool2d,softmax
   * - mobilenetv1（armv7）
     - 裁剪前
     - 967 KB
     - conv2d,depthwise_conv2d,fc,pool2d,softmax
   * - mobilenetv1（armv7）
     - 裁剪后
     - 563 KB
     - conv2d,depthwise_conv2d,fc,pool2d,softmax


实现过程：
----------

Step 1. 准备模型
^^^^^^^^^^^^^^^^


* 
  模型格式：只支持以下五种模型格式

.. code-block:: shell

     # 格式一 : __model__ + var1 + var2 + ...
     # 格式二 : model + var1 + var2 + ...
     # 格式三 : pdmodel + pdiparams
     # 格式四 : model +  params
     # 格式五 : model + weights

* 
  所有模型放入同一个文件夹

.. code-block:: bash

   # eg. 下面将 mobilenet_v1 和 shufflenet_v1 两个模型放入同一个文件夹 models
   # 假设 models 文件夹的绝对路径是 /models
   /models
    ｜- mobilenet_v1
    ｜       ｜-- model
    ｜       ｜-- params
    ｜- shufflenet_v1
             |-- __model__
             |-- var1
             |-- var1
             |-- ...

Step 2. 根据模型编译预测库
^^^^^^^^^^^^^^^^^^^^^^^^^^^

编译 Android 预测库
~~~~~~~~~~~~~~~~~~~


* 根据模型编译

.. code-block:: shell

   cd Paddle-Lite 
   ./lite/tools/build_android_by_models.sh /models
   # “模型文件夹的绝对路径” 作为脚本输入


* 编译产出

.. code-block:: shell

   # 编译产出位于： Paddle-Lite/android_lib
   android_lib  (Android 编译产出)
      |---- armv7.clang      （armv7 clang 预测库 & demo)
      |---- armv8.clang      （armv8 clang 预测库 & demo)
      |---- opt              （模型转换工具 opt)
      |---- optimized_model  （opt 转化后的 Android 移动端模型)
                 |---- mobilenet_v1.nb
                 |---- shufflenet_v1.nb


* 其他： 可以修改   ``build_android_by_models.sh`` 以改变编译选项

.. code-block:: shell

   # Paddle-Lite/lite/tools/build_android_by_models.sh

     8 WITH_LOG=OFF      # （1）可以修改 ON：运行时输出日志  OFF： 运行时不输出日志
     9 WITH_CV=ON        # （2）可以修改 ON：包含图像处理API OFF：不含图像处理 API
    10 WITH_EXCEPTION=ON # （3）可以修改 ON：DEBUG 选项（可回溯错误信息）
    11 TOOL_CHAIN=clang  #  (4) DNK 编译器： 可选择 clang 或着 gcc

编译 iOS 预测库
~~~~~~~~~~~~~~~~


* 根据模型编译

.. code-block:: shell

   cd Paddle-Lite 
   ./lite/tools/build_ios_by_models.sh --model_dir=/models
   # “模型文件夹的绝对路径” 作为脚本输入


* 编译脚本选项参数说明

.. code-block::

   --with_metal: (OFF|ON)         是否编译 iOS GPU预测库，默认为 OFF
   --with_extra: (OFF|ON)         是否编译 OCR/NLP 模型相关 kernel&OP，默认为 OFF，只编译 CV 模型相关 kernel&OP
   --with_cv: (OFF|ON)            是否编译 CV 相关预处理库, 默认为 OFF
   --with_exception: (OFF|ON)     是否在错误发生时抛出异常，默认为 OFF
   --model_dir: (Paddle 模型目录)   Paddle 模型目录，可以放多个模型，每个模型以子目录形式放置在该目录

也可以通过以下命令查看完整的参数选项

.. code-block::

   ./lite/tools/build_ios_by_models.sh help


* 编译产出

.. code-block:: shell

   # 编译产出位于： Paddle-Lite/iOS_lib
   iOS_lib  (iOS 编译产出)
      |---- armv7            （armv7 iOS 预测库 & demo)
      |---- armv8            （armv8 iOS 预测库 & demo)
      |---- opt              （模型转换工具 opt)
      |---- optimized_model  （opt 转化后的 iOS 移动端模型)
                 |---- mobilenet_v1.nb
                 |---- shufflenet_v1.nb
