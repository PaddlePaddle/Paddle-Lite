# 新增OP

以下以添加argmax为例，详细说明新增op的方法。

## 1. 添加OpParam 结构体以传导 Op 的输入和输出

- 这里命名为 `ArgmaxParam`

- 在 `paddlelite/lite/operators/op_params.h` 中添加 `ArgmaxParam` 结构体，代码如下：
    ```c++
    struct ArgmaxParam {
        lite::Tensor* X{};
        lite::Tensor* Out{};
        int Axis{0};
    };
    ```
## 2. 添加 Argmax Op 并注册

- 在paddlelite/lite/operators/目录下新建argmax_op.h文件，主要代码如下：
    ```c++
    class ArgmaxOpLite : public OpLite {
    public:
        ArgmaxOpLite() {}
        explicit ArgmaxOpLite(const std::string &op_type) : OpLite(op_type) {}
        bool CheckShape() const override;
        bool InferShape() const override;
        bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;
        void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
        std::string DebugString() const override { return "argmax"; }
    private:
        mutable ArgmaxParam param_;
    };
    ```
    `ArgmaxOpLite` 继承 `OpLite` ，成员变量包括 `ArgmaxParam` 结构体，需要实现的接口包括 `CheckShape()` 、`InferShape()` 、`AttachImp()` 、`AttachKernel()` 和 `DebugString()` 函数。`AttachKernel()` 和 `DebugString() `函数较为简单，此处直接实现；

- 在 `paddlelite/lite/operators/` 目录下新建argmax_op.cc文件，需要具体实现`CheckShape()`、`InferShape()`和`AttachImp()`函数。`CheckShape()`函数检查输入是否符合要求，`InferShape()`函数基于输入推断得到输出的维度，`AttachImp()`函数绑定Op的输入输出。然后在argmax_op.cc文件中注册argmax，核心代码如下：
    ```c++
    bool ArgmaxOpLite::CheckShape() const {
        CHECK_OR_FALSE(param_.X);
        CHECK_OR_FALSE(param_.Out);
        CHECK_OR_FALSE(param_.Axis < (param_.X)->dims().size());
        return true;
    }
    
    bool ArgmaxOpLite::InferShape() const {
        auto x_dims = param_.X->dims();
        int x_rank = x_dims.size();
        int axis = param_.Axis;
        if (axis < 0) axis += x_rank;
    
    std::vector<int64_t> out_dims;
        for (int64_t i = 0; i < axis; i++) {
            out_dims.push_back(x_dims[i]);
        }
        for (int64_t i = axis + 1; i < x_rank; i++) {
            out_dims.push_back(x_dims[i]);
        }
    
      // Set output dims
        param_.Out->Resize(lite::DDim(out_dims));
        return true;
    }
    
    bool ArgmaxOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
        auto x = op_desc.Input("X").front();
        auto out = op_desc.Output("Out").front();
    
    param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
        param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();
        param_.Axis = op_desc.GetAttr<int>("Axis");
    
    return true;
    }
    REGISTER_LITE_OP(argmax, paddle::lite::operators::ArgmaxOpLite);
    ```
- 在paddlelite/lite/operators/CMakeLists.txt中添加```add_operator(argmax_op basic SRCS argmax_op.cc DEPS ${op_DEPS})```

## 3. 添加Argmax Kernel并绑定

以下以arm端argmax实现为例说明
- 在paddlelite/lite/kernels/arm/目录下新建argmax_compute.h文件，声明ArgmaxCompute类，并继承KernelLite，主要代码如下：
    ```c++
    class ArgmaxCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
    public:
        using param_t = operators::ArgmaxParam;
        void Run() override;
        virtual ~ArgmaxCompute() = default;
    };
    ```
- 在paddlelite/lite/kernels/arm/目录下新建argmax_compute.cc文件，主要实现Run函数。`Run()`函数调用paddlelite/lite/bachends/arm/math/argmax.h中的`argmax_func()`函数，根据输入计算输出。最后在argmax_compute.cc文件中，我们绑定argmax的输入输出（为tensor的输入参数都需要绑定），代码如下：
    ```c++
    void ArgmaxCompute::Run() {
        auto& param = Param<operators::ArgmaxParam>();
        lite::Tensor* input = param.X;
        lite::Tensor* output = param.Out;
        int axis = param.Axis;
        lite::arm::math::argmax_func(input, axis, output);
        return;
    }

    REGISTER_LITE_KERNEL(
        argmax, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::ArgmaxCompute, def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
        .Finalize();
    ```

- 在paddlelite/lite/kernels/arm/CMakeLists.txt中添加
    ```cmake
    add_kernel(argmax_compute_arm ARM basic SRCS argmax_compute.cc DEPS ${lite_kernel_deps} math_arm)
    ```

## 4. 添加Argmax实现

- 在paddlelite/lite/backends/arm/math/目录下新建argmax.h文件，声明`argmax_func()`函数，代码如下：
    ```c++
    void argmax_func(const lite::Tensor* input, const int axis, lite::Tensor* output);
    ```
- 在paddlelite/lite/backends/arm/math/目录下新建argmax.cc文件，具体实现`argmax_func()`函数，代码如下：
    ```c++
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
        const float *in_ptr = input->data<float>() + n * in_channel + k;
        std::vector<std::pair<float, int>> vec;
        vec.resize(size);
        for (int i = 0; i < size; i++) {
            vec[i] = std::make_pair(in_ptr[i * in_stride], i);
        }
        // sort
        std::partial_sort(vec.begin(),
                            vec.begin() + 1,
                            vec.end(),
                            std::greater<std::pair<float, int>>());

        // out
        float *out_ptr = output->mutable_data<float>() + n * out_channel + k;
        *out_ptr = vec[0].second;
        }
    }
    }
    ```
- 在paddlelite/lite/backends/arm/math/CMakeFile.txt中的```math_arm library```中添加argmax.cc，在paddlelite/lite/backends/arm/math/funcs.h中添加```#include "lite/arm/math/argmax.h"```

## 5. 添加Argmax单测

- 在paddlelite/lite/tests/kernels目录下新建argmax_compute_test.cc文件，声明并实现ArgmaxComputeTester类；
- ArgmaxComputeTester类中主要包括PrepareOpDesc、PrepareData和RunBaseline函数。PrepareOpDesc函数设定单测op的类型和输入输出参数，PrepareData函数对输入tensor进行初始化，RunBaseline是基于输入计算得到输出，用于和框架计算的输出进行对比；
- 使用gtest添加单测，代码如下：
    ```c++
    TEST(Argmax, precision) {
        #ifdef LITE_WITH_ARM
        LOG(INFO) << "test argmax arm";
        Place place(TARGET(kARM));

        for (int axis : {0, 1, 2, 3}) {
            for (int n : {1, 3}) {
            for (int c : {3, 6}) {
                for (int h : {9, 18}) {
                for (int w : {9, 18}) {
                    std::unique_ptr<arena::TestCase> tester(
                        new ArgmaxComputeTester(place, "def", axis, n, c, h, w));
                    arena::Arena arena(std::move(tester), place, 2e-5);
                    arena.TestPrecision();
                }
                }
            }
            }
        }
        #endif
    }
    ```
- 在paddlelite/lite/tests/kernels/CMakeLists.txt中添加
    ```cmake
    lite_cc_test(test_kernel_argmax_compute SRCS argmax_compute_test.cc DEPS arena_framework ${x86_kernels} ${arm_kernels} ${lite_ops} ${host_kernels})
    ```
## 6. 编译运行
- 在paddlelite目录中，执行```./lite/tools/ci_build.sh build_test_arm```，该脚本会创建手机模拟器，并编译运行所有单测（花费时间较久）。如果运行无误，则表明添加argmax成功。
