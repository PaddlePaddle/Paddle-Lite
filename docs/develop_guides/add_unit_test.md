# 新增单测

本文主要介绍如何添加 ARM CPU/GPU（OpenCL 和 Metal） 和 Host 后端 OP 单测实现。以添加 Argmax 为例，将详细说明新增单测的方法。

## 1. 添加 Argmax 单测

目前有如下 2 种方式，其中基于 Autoscan 框架实现的 Python 单测代码具有覆盖度高、代码量少、支持与 Paddle 原生精度对齐等优点，因此推荐使用该方式。

- 基于 Autoscan 框架，实现 Python 单测代码
- 基于 gtest，实现 C++ 单测单码

### 1.1 Python 单测

Python 单测测试方法：通过 `sample_program_configs` 方法定义 OP 的输入 shape 和属性信息，并构建出一个网络；然后通过 `sample_predictor_configs` 方法确定运行后端的 config 信息；最好通过 `test` 方法，完成单测测试。精度对比方法：将 Paddle Lite 的输出结果和 PaddlePaddle 的输出结果进行比较，判断两者绝对误差和相对误差大小，以确定单测的正确性。

在 Paddle-Lite/lite/tests/unittest_py/op 目录下新建 [test_arg_max_op.py](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/tests/unittest_py/op/test_arg_max_op.py) 文件，定义 `TestArgMaxOp` 类，其继承自 `AutoScanTest`，重点介绍如下 4 个方法：

- `__init__` 方法设置 Place 属性，通过调用 `self.enable_testing_on_place` 方法激活特定后端；
- `is_program_valid` 方法用于 Op 属性和输入的合法性规则检查；
- `sample_program_configs` 方法定义输入 shape、输入数据类型、参数信息等，生成 program_config；
- `sample_predictor_configs` 方法返回 paddlelite_configs, op_list 和误差阈值；
- `add_ignore_pass_case` 方法设置一些当前实现运行错误的情况需要跳过的情况;
- `test` 方法为 unittest 的运行入口函数。

#### 1.1.1 `AutoScanTest` 类 `__init__` 方法

该方法用于初始化设备信息，明确单测在哪个后端运行。通过调用 `self.enable_testing_on_place` 方法激活特定后端，并提供精度信息、layout 信息、线程信息等相关信息的配置。

    ```python
        # enable_testing_on_place 方法，无返回值
        def enable_testing_on_place(self,
                                        target=None,
                                        precision=None,
                                        layout=None,
                                        thread=None,
                                        places=None) -> None:
        # 单个 place 设置，默认选择 FP32 kernel 计算
        self.enable_testing_on_place(TargetType.Host,
                    PrecisionType.FP32,
                    DataLayoutType.NCHW,
                    thread=[1, 4])
        # 多个 place 设置, kernel 选择根据 arm_place 里的每个place 值进行一一匹配，直到匹配成功为止
        # 具体来说：如果这个 OP 支持 int8 计算，则优先选用 int8 kernel 计算
        arm_place = [
                    Place(TargetType.ARM, PrecisionType.INT8, DataLayoutType.NCHW),
                    Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)
                ]
        self.enable_testing_on_place(places=arm_place)
    ```


#### 1.1.2 `AutoScanTest` 类 `sample_program_configs` 方法

该方法通过设置输入信息（如 OP 属性、输入 shape 等），生成一个网络，用于推理。详细使用方法如下所示：

   ```python
        # sample_program_configs 方法
        def sample_program_configs(self, *args, **kwargs):
        # 返回值：返回一个网络信息 program_config

        # 以 assign OP 为例，根据 assign OP 的属性，构建其需要的输入信息，并生成网络
        def sample_program_configs(self, *args, **kwargs):
            # step1. 用函数来定义输入数据
            def generate_input(*args, **kwargs):
                return np.random.random(in_shape).astype(np.float32)
            # Step2. 定义OP, 名称\输入\输出\属性
            # 通过 import hypothesis.strategies as st 随机生成输入 shape 大小
            # draw 方法用于采样，选取某些 case 进行测试
            in_shape = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=8), max_size=4))
            # OpConfig 用于创建 OP，完成 OP 的输入、输出和属性的配置
            assign_op = OpConfig(
                type = "assign",
                inputs = {"X" : ["input_data"]},
                outputs = {"Out": ["output_data"]},
                attrs = {})
            # Step3. 将数据和 OP 定义联系起来
            program_config = ProgramConfig(
                ops=[assign_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input, *args, **kwargs)),
                },
                outputs=["output_data"])
            yield program_config
   ```

#### 1.1.3 `AutoScanTest` 类 `is_program_valid` 方法

该方法用于判读当前输入 case 下，这个网络是否有效。
**注意：**
>> 不合理的判定标准为：paddle 组网时报错。此外，过滤功能鼓励优先在 `sample_program_config` 函数中使用 assume 接口过滤。

    ```python
        # is_program_valid 方法
        def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        # 参数：program_config 存放当前网络结构
        # 参数：predictor_config 存放当前网络运行后端设备的配置信息
        # 返回值：True 表示当前网络结构有效；False 表示当前网络结构无效

        # assign OP 中的 is_sparse 属性必须为true，如若出现false，则该网络结构无效
        # is_sparse is only support False
        class TestAssignOp(AutoScanTest):
            def is_program_valid(self, program_config: ProgramConfig) -> bool:
            if program_config.ops[0].attrs['is_sparse'] == True:
               return False
            return True
    ```

#### 1.1.4 `AutoScanTest` 类 `sample_predictor_configs` 方法

该方法用于配置测试后端 config 信息，确定网络在哪个后端运行。

    ```python
        # sample_predictor_configs 方法
        def sample_predictor_configs(self, program_config):
        # 参数 program_config ：存放当前网络结构
        # 返回值：运行后端config 信息，OP 名字 和 误差大小list[绝对误差，相对误差]

        # 以 assign OP 为例，让 OP 在 ARM 端运行
        def sample_program_configs(self, *args, **kwargs):
            # 方法一：通过 get_predictor_configs 直接获取 __init__ 方法中的place信息，极力推荐
            return self.get_predictor_configs(), ["assign"], (1e-5, 1e-5)
            # 方法二：通过 CXXConfig 手动配置
            # Step1. 执行后端是arm、线程数 1
            config = CxxConfig()
            config.set_valid_places({Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)})
            config.set_threads(1)
            # Step2. 相对误差/绝对误差 上限 = (1e-5, 1e-5)
            yield config, ["assign"], (1e-5, 1e-5)
   ```

#### 1.1.5 `AutoScanTest` 类 `add_ignore_pass_case` 方法

该方法用于跳过某些测试 case，并完成标记和记录下个版本的修复内容。当前 `IgnoreReasonsBase` 支持以下过滤类型的单测案例：

  - PADDLE_NOT_SUPPORT : 组网成功，但 Paddle 推理过程中报错导致进程退出。即组网成功，Paddle 与 Paddle Lite 均不推理

  - PADDLELITE_NOT_SUPPORT : Paddle Lite 没有对应算子或者 Paddle Lite 推理时报错导致进程退出。即组网成功，Paddle 推理但 Paddle Lite 不推理

  - ACCURACY_ERROR : Paddle 与 Paddle Lite计算结果存在 diff。即组网成功，Paddle 和 Paddle Lite 均推理，但不对比输出结果

```python
        # add_ignore_pass_case 方法
        def add_ignore_pass_case(self):
        # 返回值：True 表示过滤测试 case，不做精度测试，False 表示不过滤，需要完成精度测试
        # 首先，通过定义 _teller(program_config, predictor_config) 函数，完成过滤 case 的书写
        # 然后，通过self.add_ignore_check_case(IgnoreReasonsBase, note) 将过滤函数加入过滤集

        # 以 conv2d OP 为例，Metal 不支持 groups !=1 的计算，后续会补充支持
        def add_ignore_pass_case(self):
           def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            input_shape = program_config.inputs["input_data"].shape
            filter_data = program_config.weights["filter_data"].shape
            groups = program_config.ops[0].attrs["groups"]
            if target_type == TargetType.Metal:
                if groups != 1:
                    return True

            self.add_ignore_check_case(
                _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
                "Lite does not support this op in a specific case on metal. We need to fix it as soon as possible."
            )
   ```

#### 1.1.6 `AutoScanTest` 类 `test` 方法

该方法是 OP 单测方法的测试入口函数，完成 OP 单测测试。

    ```python
      # test 方法
      def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)
      # 参数：quant 用于判断是否量化，当前不使用。（单测跟据 predict_config 的 place 信息进行判断）
      # 参数：max_examples 表示测试次数，即随机采样 max_examples 次数，进行测试
    ```

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
