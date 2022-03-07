# 新增单测

本文主要介绍如何添加 ARM CPU/GPU（OpenCL 和 Metal） 和 Host 后端 OP 单测实现。以添加 Argmax 为例，将详细说明新增单测的方法。

## 1. 添加 Argmax 单测

目前有如下 2 种方式，其中基于 Autoscan 框架实现的 Python 单测代码具有覆盖度高、代码量少、支持与 Paddle 原生精度对齐等优点，因此推荐使用该方式。

- 基于 Autoscan 框架，实现 Python 单测代码
- 基于 gtest，实现 C++ 单测单码

### 1.1 Python 单测

在 Paddle-Lite/lite/tests/unittest_py/op 目录下新建 [test_arg_max_op.py](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tests/unittest_py/op/test_arg_max_op.py) 文件，定义 `TestArgMaxOp` 类，其继承自 `AutoScanTest`，重点介绍如下 4 个方法：

- `__init__` 方法设置 Place 属性，通过调用 `self.enable_testing_on_place` 方法激活特定后端；
- `is_program_valid` 方法用于 Op 属性和输入的合法性规则检查；
- `sample_program_configs` 方法定义输入 shape、输入数据类型、参数信息等，生成 program_config；
- `sample_predictor_configs` 方法返回 paddlelite_configs, op_list 和误差阈值；
- `add_ignore_pass_case` 方法设置一些当前实现运行错误的情况需要跳过的情况;
- `test` 方法为 unittest 的运行入口函数。

### 1.2 C++ 单测

以 ARM CPU 为例：

- 在 Paddle-Lite/lite/tests/kernels 目录下新建 [argmax_compute_test.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tests/kernels/argmax_compute_test.cc) 文件，声明并实现 `ArgmaxComputeTester` 类；

- `ArgmaxComputeTester` 类中主要包括 `PrepareOpDesc`、`PrepareData` 和 `RunBaseline` 函数。`PrepareOpDesc` 函数设定单测 Op 的类型和输入输出参数，`PrepareData` 函数对输入 Tensor 进行初始化，`RunBaseline` 是基于输入计算得到输出，用于和框架计算的输出进行对比；

- 使用 gtest 添加单测，代码如下：

    ```c++
    void TestArgmax(const Place& place) {
        for (int axis : {-1, -2, 0, 2}) {
            for (bool keepdims : {false, true}) {
                for (int dtype : {-1, 2, 3}) {
                    for (auto x_shape :
                        std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {2, 3, 4, 5}}) {
                        int x_size = x_shape.size();
                        if (axis < -x_size || axis >= x_size) continue;
    #if defined(LITE_WITH_NNADAPTER)
                        std::vector<std::string> alias_vec{"def"};
    #else
                        std::vector<std::string> alias_vec{
                        "fp32", "int64", "int32", "int16", "uint8"};
    #endif
                        for (std::string alias : alias_vec) {
                            std::unique_ptr<arena::TestCase> tester(new ArgmaxComputeTester(
                                place, alias, axis, keepdims, dtype, DDim(x_shape)));
                            arena::Arena arena(std::move(tester), place, 2e-5);
                            arena.TestPrecision();
                        }
                    }
                }
            }
        }
    }

    TEST(Argmax, precision) {
        Place place;
    #if defined(LITE_WITH_NNADAPTER) && defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
        place = TARGET(kNNAdapter);
    #elif defined(LITE_WITH_ARM)
        place = TARGET(kARM);
    #elif defined(LITE_WITH_X86)
        place = TARGET(kHost);
    #else
        return;
    #endif

        TestArgmax(place);
    }
    ```

- 在 Paddle-Lite/lite/tests/kernels/CMakeLists.txt 中添加
    ```lite_cc_test(test_kernel_argmax_compute SRCS argmax_compute_test.cc))```

## 2. 编译运行

## 2.1 Python 单测

### 2.1.1 硬件环境

- 配置苹果 M1 芯片的机器：适用于验证 ARM, OpenCL, Metal 后端的单测
- 配置 Intel 芯片的 Linux 机器：适用于验证 Host, X86 后端的单测

### 2.1.2 软件环境

#### 2.1.2.1 针对 ARM, OpenCL, Metal 后端

- M1 设备上需安装两种版本的 Python 环境
    - `python3.8 Intel` （下载并安装）版本：[3.8.10 intel](https://www.python.org/ftp/python/3.8.10/python-3.8.10-macosx10.9.pkg)
    - `python3.9 ARM`（下载并安装） 版本：[3.9.8 universal](https://www.python.org/ftp/python/3.9.8/python-3.9.8-macos11.pkg)

- 创建并激活 Python 虚拟环境
	- ```python3.9 -m venv ${your custiom python path}```
	- ```source ${your custom python path}/bin/activate```

- 安装依赖项
    - `cd Paddle-Lite/lite/tests/unittest_py/ && python3.8 -m pip install -r requirements.txt`
    - `cd Paddle-Lite/lite/tests/unittest_py/rpc_service && python3.9 -m pip install -r requirements.txt`

- 安装 `Paddle`
    - `python3.8 -m pip install paddlepaddle`

- 源码编译 whl 包并安装
    - `./lite/tools/build_macos.sh --with_python=ON --with_opencl=ON --with_metal=ON --with_arm82_fp16=ON --python_version=3.9 arm64 && python3.9 -m pip install --force-reinstall $(find ./build.macos.armmacos.armv8.* -name *whl)`

- 运行
    - ARM CPU: `cd lite/tests/unittest_py/op && ./auto_scan.sh test_arg_max_op.py --target=ARM`
    - OpenCL: `cd lite/tests/unittest_py/op && ./auto_scan.sh test_arg_max_op.py --target=OpenCL`
    - Metal: `cd lite/tests/unittest_py/op && ./auto_scan.sh test_arg_max_op.py --target=Metal`

#### 2.1.2.2 针对 Host, X86 后端
- Linux 机器（推荐 Ubuntu 18.04）上安装 Python3.7
    - `sudo apt-get install python==3.7`

- 安装依赖项
    - `cd Paddle-Lite/lite/tests/unittest_py/ && python3.7 -m pip install -r requirements.txt`

- 安装 `Paddle`
    - `python3.7 -m pip install paddlepaddle`

- 源码编译 whl 包并安装
    - `./lite/tools/build_linux.sh --with_python=ON --python_version=3.7 --with_extra=ON --arch=x86 && python3.7 -m pip install --force-reinstall $(find ./build.lite.linux.x86.* -name *whl)`

- 运行
    - Host: `cd lite/tests/unittest_py/op && ./auto_scan.sh test_arg_max_op.py --target=Host`
    - X86: `cd lite/tests/unittest_py/op && ./auto_scan.sh test_arg_max_op.py --target=X86`

## 2.2 C++ 单测

- 在 Paddle-Lite 目录中，执行 ```./lite/tools/ci_build.sh build_test_arm```，该脚本会创建手机模拟器，并编译运行所有单测（花费时间较久）。如果运行无误，则表明添加 Argmax 成功。
