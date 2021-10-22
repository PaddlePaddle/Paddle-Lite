# 新增OP

以下以添加 Argmax 为例，详细说明新增 Op 的方法。

## 1. 添加 OpParam 结构体以传导 Op 的输入和输出

- 这里命名为 `ArgmaxParam`

- 在 `paddlelite/lite/operators/op_params.h` 中添加 `ArgmaxParam` 结构体，代码如下：
    ```c++
    struct ArgmaxParam : ParamBase {
      lite::Tensor* X{};
      lite::Tensor* Out{};
      int Axis{0};
      int dtype{-1};
      bool keepdims{false};
    };
    ```
## 2. 添加 Argmax Op 并注册

- 在 paddlelite/lite/operators/ 目录下新建 argmax_op.h 文件，主要代码如下：
    ```c++
    class ArgmaxOpLite : public OpLite {
    public:
        ArgmaxOpLite() {}
        explicit ArgmaxOpLite(const std::string &op_type) : OpLite(op_type) {}
        bool CheckShape() const override;
        bool InferShapeImpl() const override;
        bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;
        void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
        std::string DebugString() const override { return "argmax"; }

    #ifdef LITE_WITH_PROFILE
        void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
            auto input_dims = param_.X->dims();
            auto output_dims = param_.Out->dims();
            ch->input_shape = ch->DimToStr(input_dims);
            ch->output_shape = ch->DimToStr(output_dims);
            ch->remark = "axis" + std::to_string(param_.Axis);

            auto axis = param_.Axis;
            if (axis < 0) {
                axis += input_dims.size();
            }
            int max_num = 1;
            for (int64_t i = axis + 1; i < input_dims.size(); i++)
                max_num *= input_dims[i];
            float gops = 1.0f;
            for (int i = 1; i <= max_num; i++) gops *= i;
            ch->macs = gops * output_dims.production();
        }
    #endif

    private:
        mutable ArgmaxParam param_;
    };
    ```
    `ArgmaxOpLite` 继承 `OpLite` ，成员变量包括 `ArgmaxParam` 结构体，需要实现的接口包括 `CheckShape()` 、`InferShapeImpl()` 、`AttachImpl()` 、`AttachKernel()` 和 `DebugString()` 函数。`AttachKernel()` 和 `DebugString()` 函数较为简单，此处直接实现；

- 在 `paddlelite/lite/operators/` 目录下新建 argmax_op.cc 文件，需要具体实现 `CheckShape()`、`InferShapeImpl()` 和 `AttachImpl()` 函数。`CheckShape()` 函数检查输入是否符合要求，`InferShape()` 函数基于输入推断得到输出的维度，`AttachImpl()` 函数绑定 Op 的输入输出。然后在 argmax_op.cc 文件中注册 Argmax，核心代码如下：
    ```c++
    bool ArgmaxOpLite::CheckShape() const {
        CHECK_OR_FALSE(param_.X);
        CHECK_OR_FALSE(param_.Out);
        CHECK_OR_FALSE(param_.Axis < static_cast<int>((param_.X)->dims().size()));
        CHECK_OR_FALSE(param_.Axis >= static_cast<int>(-(param_.X)->dims().size()));
        return true;
    }
    
    bool ArgmaxOpLite::InferShapeImpl() const {
        auto x_dims = param_.X->dims();
        int x_rank = x_dims.size();
        int axis = param_.Axis;
        if (axis < 0) {
          axis += x_rank;
        }

        std::vector<int64_t> out_dims;
        for (int64_t i = 0; i < axis; i++) out_dims.push_back(x_dims[i]);
        if (param_.keepdims) {
            out_dims.push_back(static_cast<int64_t>(1));
        }
        for (int64_t i = axis + 1; i < x_rank; i++) out_dims.push_back(x_dims[i]);
        // Set output dims
        param_.Out->Resize(lite::DDim(out_dims));
        return true;
    }
    
    bool ArgmaxOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
        auto x = op_desc.Input("X").front();
        auto out = op_desc.Output("Out").front();

        if (op_desc.HasAttr("keepdims")) {
            param_.keepdims = op_desc.GetAttr<bool>("keepdims");
        }
        if (op_desc.HasAttr("dtype")) {
            param_.dtype = op_desc.GetAttr<int>("dtype");
        }

        param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
        param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();
        param_.Axis = op_desc.GetAttr<int64_t>("axis");

        return true;
    }
    REGISTER_LITE_OP(arg_max, paddle::lite::operators::ArgmaxOpLite);
    ```
- 在 paddlelite/lite/operators/CMakeLists.txt 中添加 ```add_operator(argmax_op basic SRCS argmax_op.cc)```

## 3. 添加 Argmax Kernel 并绑定

以下以 Arm 端 Argmax 实现为例说明
- 在 paddlelite/lite/kernels/arm/ 目录下新建 argmax_compute.h 文件，声明 ArgmaxCompute 类，并继承 KernelLite，主要代码如下：
    ```c++
    template <typename T>
    class ArgmaxCompute : public KernelLite<TARGET(kARM), PRECISION(kAny)> {
    public:
        using param_t = operators::ArgmaxParam;
        void Run() override;
        virtual ~ArgmaxCompute() = default;
    #ifdef LITE_WITH_PROFILE
        virtual void SetProfileRuntimeKernelInfo(
            paddle::lite::profile::OpCharacter* ch) {
            ch->kernel_func_name = kernel_func_name_;
        }
        std::string kernel_func_name_{"NotImplForArgmax"};
    #endif
    };
    ```
- 在 paddlelite/lite/kernels/arm/ 目录下新建 argmax_compute.cc 文件，主要实现 Run 函数。`Run()` 函数调用 paddlelite/lite/bachends/arm/math/argmax.h 中的 `argmax_func()` 函数，根据输入计算输出。最后在 argmax_compute.cc 文件中，我们绑定 Argmax 的输入输出（为 Tensor 的输入参数都需要绑定），代码如下：
    ```c++
    template <typename T>
    void ArgmaxCompute<T>::Run() {
        auto& param = Param<operators::ArgmaxParam>();
        lite::Tensor* input = param.X;
        lite::Tensor* output = param.Out;
        int axis = param.Axis;
        if (axis < 0) {
            axis += input->dims().size();
        }

        switch (param.dtype) {
            // default indices type: int64_t
            case -1: {
                lite::arm::math::argmax_func<T, int64_t>(input, axis, output);
                break;
            }
            // static_cast<int>(lite::core::FluidType::INT32) == 2
            case 2: {
                lite::arm::math::argmax_func<T, int32_t>(input, axis, output);
                break;
            }
            // static_cast<int>(lite::core::FluidType::INT64) == 3
            case 3: {
                lite::arm::math::argmax_func<T, int64_t>(input, axis, output);
                break;
            }
            default: {
                LOG(FATAL) << "Attribute `dtype` in arg_max op must be 2 or 3, which "
                              "indicates that indices dtype must be int32 or int64, "
                              "default dtype is int64.";
                break;
            }
        }
    #ifdef LITE_WITH_PROFILE
        kernel_func_name_ = "argmax_func";
    #endif
        return;
    }

    REGISTER_LITE_KERNEL(arg_max,
                         kARM,
                         kAny,
                         kNCHW,
                         paddle::lite::kernels::arm::ArgmaxCompute<float>,
                         fp32)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
        .BindPaddleOpVersion("arg_max", 1)
        .Finalize();
    ```

- 在 paddlelite/lite/kernels/arm/CMakeLists.txt 中添加
    ```add_kernel(argmax_compute_arm ARM basic SRCS argmax_compute.cc)```

## 4. 添加 Argmax 实现

- 在 paddlelite/lite/backends/arm/math/ 目录下新建 argmax.h 文件，声明 `argmax_func()` 函数，代码如下：
    ```c++
    template <typename InType, typename OutType>
    void argmax_func(const lite::Tensor* input,
                     const int axis,
                     lite::Tensor* output);
    ```
- 在 paddlelite/lite/backends/arm/math/ 目录下新建 argmax.cc 文件，具体实现 `argmax_func()` 函数，代码如下：
    ```c++
    template <typename InType, typename OutType>
    void argmax_func(const lite::Tensor *input,
                     const int axis,
                     lite::Tensor *output) {
        auto input_ddim = input->dims();
        auto output_ddim = output->dims();

        const int size = input_ddim[axis];
        const int in_channel = input_ddim.count(axis, input_ddim.size());
        const int out_channel = output_ddim.count(axis, output_ddim.size());
        const int in_stride = input_ddim.count(axis + 1, input_ddim.size());
        const int out_stride = input_ddim.count(0, axis);

        for (int n = 0; n < out_stride; n++) {
            for (int k = 0; k < in_stride; k++) {
                const InType *in_ptr = input->data<InType>() + n * in_channel + k;
                std::vector<std::pair<InType, OutType>> vec;
                vec.resize(size);
                for (int i = 0; i < size; i++) {
                    vec[i] = std::make_pair(in_ptr[i * in_stride], i);
                }
                // sort
                std::partial_sort(vec.begin(),
                                  vec.begin() + 1,
                                  vec.end(),
                                  std::greater<std::pair<InType, OutType>>());
                // out
                OutType *out_ptr = output->mutable_data<OutType>() + n * out_channel + k;
                *out_ptr = vec[0].second;
            }
        }
    }
    ```
- 在 paddlelite/lite/backends/arm/math/CMakeFile.txt 中的 ```math_arm library``` 中添加 argmax.cc，在 paddlelite/lite/backends/arm/math/funcs.h 中添加 ```#include "lite/backends/arm/math/argmax.h"```

## 5. 添加 Argmax 单测

- 在 paddlelite/lite/tests/kernels 目录下新建 argmax_compute_test.cc 文件，声明并实现 ArgmaxComputeTester 类；
- ArgmaxComputeTester 类中主要包括 PrepareOpDesc、PrepareData 和 RunBaseline 函数。PrepareOpDesc 函数设定单测 Op 的类型和输入输出参数，PrepareData 函数对输入 Tensor 进行初始化，RunBaseline 是基于输入计算得到输出，用于和框架计算的输出进行对比；
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
- 在 paddlelite/lite/tests/kernels/CMakeLists.txt 中添加
    ```lite_cc_test(test_kernel_argmax_compute SRCS argmax_compute_test.cc))```
## 6. 编译运行
- 在 paddlelite 目录中，执行 ```./lite/tools/ci_build.sh build_test_arm```，该脚本会创建手机模拟器，并编译运行所有单测（花费时间较久）。如果运行无误，则表明添加 Argmax 成功。
