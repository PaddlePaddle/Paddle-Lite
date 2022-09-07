# 新增 OP

本文主要介绍如何添加 ARM CPU/GPU（OpenCL 和 Metal） 和 Host 后端 OP 算子实现。以添加 Argmax 为例，将详细说明新增 Op 的方法。

## 1. 添加 OpParam 结构体以传导 Op 的输入和输出

- 这里命名为 `ArgmaxParam`

- 在 `Paddle-Lite/lite/operators/op_params.h` 中添加 `ArgmaxParam` 结构体，代码如下：
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

- 在 Paddle-Lite/lite/operators/ 目录下新建 [argmax_op.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/operators/argmax_op.h) 文件，主要代码如下：
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

- 在 `Paddle-Lite/lite/operators/` 目录下新建 [argmax_op.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/operators/argmax_op.cc) 文件，需要具体实现 `CheckShape()`、`InferShapeImpl()` 和 `AttachImpl()` 函数。`CheckShape()` 函数检查输入是否符合要求，`InferShape()` 函数基于输入推断得到输出的维度，`AttachImpl()` 函数绑定 Op 的输入输出。然后在 argmax_op.cc 文件中注册 Argmax，核心代码如下：
    
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

- 在 Paddle-Lite/lite/operators/CMakeLists.txt 中添加 ```add_operator(argmax_op basic SRCS argmax_op.cc)```

## 3. 添加 Argmax Kernel 并绑定

Paddle Lite 有 Host, ARM, x86, OpenCL, Metal, NNAdapter 等多种后端，同一 Op 在不同后端的代码实现细节不同，因此需要具体讨论。

### 3.1 Host 端

Host 端算子用于添加无优化实现的算子，可以在各个硬件平台运行的算子。现以 Host 端 Argmax 实现为例说明：
- 在 Paddle-Lite/lite/kernels/host/ 目录下新建 [argmax_compute.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/host/activation_compute.h) 文件，声明 ArgmaxCompute 类，并继承 KernelLite，主要代码如下：

    ```c++
    template <typename T>
    class ArgmaxCompute : public KernelLite<TARGET(kHost), PRECISION(kAny)> {
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

- 在 Paddle-Lite/lite/kernels/host/ 目录下新建 [argmax_compute.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/host/activation_compute.cc) 文件，主要实现 Run 函数。`Run()` 函数调用 Paddle-Lite/lite/backends/host/math/argmax.h 中的 `argmax_func()` 函数，根据输入计算输出。最后在 argmax_compute.cc 文件中，我们绑定 Argmax 的输入输出（为 Tensor 的输入参数都需要绑定），代码如下：

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
                         kHost,
                         kAny,
                         kNCHW,
                         paddle::lite::kernels::host::ArgmaxCompute<float>,
                         fp32)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
        .BindPaddleOpVersion("arg_max", 1)
        .Finalize();
    ```

- 在 Paddle-Lite/lite/kernels/host/CMakeLists.txt 中添加
    ```add_kernel(argmax_compute_arm Host basic SRCS argmax_compute.cc)```

### 3.2 ARM 端

以 ARM 端 Argmax 实现为例说明：
- 在 Paddle-Lite/lite/kernels/arm/ 目录下新建 [argmax_compute.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/arm/activation_compute.h) 文件，声明 ArgmaxCompute 类，并继承 KernelLite，主要代码如下：

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

- 在 Paddle-Lite/lite/kernels/arm/ 目录下新建 [argmax_compute.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/arm/activation_compute.cc) 文件，主要实现 Run 函数。`Run()` 函数调用 Paddle-Lite/lite/backends/arm/math/argmax.h 中的 `argmax_func()` 函数，根据输入计算输出。最后在 argmax_compute.cc 文件中，我们绑定 Argmax 的输入输出（为 Tensor 的输入参数都需要绑定），代码如下：

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

- 在 Paddle-Lite/lite/kernels/arm/CMakeLists.txt 中添加
    ```add_kernel(argmax_compute_arm ARM basic SRCS argmax_compute.cc)```

### 3.3 OpenCL 端

以 OpenCL 端 Argmax 实现为例说明：
- 在 Paddle-Lite/lite/kernels/opencl/ 目录下新建 [argmax_image_compute.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/opencl/argmax_image_compute.cc) 文件，定义 ArgmaxComputeImage2D 类，并继承 KernelLite，ArgmaxComputeImage2D 类主要代码如下：

    ```c++
    class ArgmaxComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                                   PRECISION(kFP16),
                                                   DATALAYOUT(kImageDefault)> {
     public:
      using param_t = operators::ArgmaxParam;
      void PrepareForRun() override;
      void ReInitWhenNeeded() override;
      void Run() override;
    #ifdef LITE_WITH_PROFILE
      void SetProfileRuntimeKernelInfo(
          paddle::lite::profile::OpCharacter* ch) override;
    #endif
    };
    ```

    重点介绍如下 4 个功能函数：
    - `PrepareForRun` 函数只在第一次运行时执行，主要功能为确定运行所需的参数、kernel 名字、编译 kernel 代码；
    - `ReInitWhenNeeded` 函数只在第一次运行时和输入 shape 发生变化时执行，主要功能为根据 shape 信息确定线程分配；
    - `Run` 函数在每次运行时均执行，主要功能为分配/获取 tensor 数据、执行 cl kernel 函数；
    - `SetProfileRuntimeKernelInfo` 函数用于 profile。

- 在 Paddle-Lite/lite/kernels/opencl/CMakeLists.txt 中添加
    ```add_kernel(argmax_opencl_image OPENCL basic SRCS argmax_image_compute.cc)```

### 3.4 Metal 端

以 Metal 端 Argmax 实现为例说明：
- 在 Paddle-Lite/lite/kernels/metal/image_op 目录下新建 [argmax_image_compute.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/metal/image_op/argmax_image_compute.h) 文件，定义 ArgmaxImageCompute 类，并继承 KernelLite，ArgmaxImageCompute 类主要代码如下：

    ```c++
    class ArgmaxImageCompute
        : public KernelLite<TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray)> {
        using param_t = operators::ArgmaxParam;

       public:
        void PrepareForRun() override;
        void Run() override;
        void SaveOutput() override {
            MetalDebug::SaveOutput((use_mps_ ? ("MPS_argmax") : function_name_), output_buffer_);
        };
        virtual ~ArgmaxImageCompute();
    };
    ```

    重点介绍如下 3 个功能函数：
    - `PrepareForRun` 函数只在第一次运行时执行，主要功能为确定运行所需的参数、kernel 名字、编译 kernel 代码；
    - `Run` 函数在每次运行时均执行，主要功能为分配/获取 tensor 数据、执行 Metal kernel 函数；
    - `SaveOutput` 函数用于 Metal 每层结果的输出。

- 在 Paddle-Lite/lite/kernels/metal/image_op 目录下新建 argmax_image_compute.mm 文件，主要实现 PrepareForRun、Run 函数，代码如下：

    ```c++
    void ArgmaxImageCompute::PrepareForRun() {
        auto& context = ctx_->As<MTLContext>();
        metal_context_ = (MetalContext*)context.context();

        const auto& param = this->Param<param_t>();
        auto output_dims = param.Out->dims();

    #ifdef LITE_WITH_METAL_FULL
    #else
        input_buffer_ = param.X->data<MetalHalf, MetalImage>();
        output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(
            metal_context_, MetalImage::FourDimFrom(output_dims));
    #endif

        // use mps or not
        bool should_use_mps = false;
        if (@available(iOS 12.0, *)) {
            if (metal_context_->use_mps()) {
                if( param.Axis == 1) should_use_mps = true;
            }
        }
        use_mps_ = should_use_mps;
        if (use_mps_) {
            setup_with_mps();
        } else {
            setup_without_mps();
        }
    }

    void ArgmaxImageCompute::Run() {
        @autoreleasepool {
            if (use_mps_) {
                run_with_mps();
            } else {
                run_without_mps();
            }
        }
    }

    REGISTER_LITE_KERNEL(arg_max,
        kMetal,
        kFloat,
        kMetalTexture2DArray,
        paddle::lite::kernels::metal::ArgmaxImageCompute,
        Int32)
        .BindInput("X",
            {LiteType::GetTensorTy(TARGET(kMetal),
                PRECISION(kFloat),
                DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out",
            {LiteType::GetTensorTy(TARGET(kMetal),
                PRECISION(kInt64),
                DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();
    ```

    Metal kernel 的实现方式有两种分别为 MPS 和非 MPS，在 PrepareForRun 函数中通过 use_mps_ 来判断具体采用的实现方式。

- 在 Paddle-Lite/lite/kernels/metal/CMakeLists.txt 中添加
    ```add_kernel(argmax_metal_image METAL basic SRCS image_op/argmax_image_compute.mm)```

## 4. 添加 Argmax 实现

### 4.1 Host 端

- 在 Paddle-Lite/lite/backends/host/math/ 目录下新建 [argmax.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/backends/host/math/argmax.h) 文件，声明 `argmax_func()` 函数，代码如下：

    ```c++
    template <typename InType, typename OutType>
    void argmax_func(const lite::Tensor* input,
                     const int axis,
                     lite::Tensor* output);
    ```

- 在 Paddle-Lite/lite/backends/host/math/ 目录下新建 [argmax.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/backends/host/math/argmax.cc) 文件，具体实现 `argmax_func()` 函数，代码如下：

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

- 在 Paddle-Lite/lite/backends/host/math/CMakeLists.txt 中的 ```math_host library``` 中添加 argmax.cc，在 Paddle-Lite/lite/backends/host/math/funcs.h 中添加 ```#include "lite/backends/host/math/argmax.h"```

### 4.2 ARM 端

- 在 Paddle-Lite/lite/backends/arm/math/ 目录下新建 [argmax.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/backends/arm/math/argmax.h) 文件，声明 `argmax_func()` 函数，代码如下：
    ```c++
    template <typename InType, typename OutType>
    void argmax_func(const lite::Tensor* input,
                     const int axis,
                     lite::Tensor* output);
    ```
- 在 Paddle-Lite/lite/backends/arm/math/ 目录下新建 [argmax.cc](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/backends/arm/math/argmax.cc) 文件，具体实现 `argmax_func()` 函数，代码如下：
    
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

- 在 Paddle-Lite/lite/backends/arm/math/CMakeLists.txt 中的 ```math_arm library``` 中添加 argmax.cc，在 Paddle-Lite/lite/backends/arm/math/funcs.h 中添加 ```#include "lite/backends/arm/math/argmax.h"```

### 4.3 OpenCL 端

- 在 Paddle-Lite/lite/backends/opencl/cl_kernel/image/ 目录下新建 [argmax_kernel.cl](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/backends/opencl/cl_kernel/image/argmax_kernel.cl) 文件，定义具体的 cl kernel 函数。

### 4.4 Metal 端

- 在 Paddle-Lite/lite/backends/metal/metal_kernel/texture/ 目录下新建 [MaxKernel.metal](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/backends/metal/metal_kernel/texture/MaxKernel.metal) 文件，定义具体的 arg_max_c 函数，其中输入的数据格式为 texture2d_array 。

    ```c++
    kernel void arg_max_c(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
        texture2d_array<ftype, access::write> outTexture[[texture(1)]],
        constant ArgParam& param[[buffer(0)]],
        uint3 gid[[thread_position_in_grid]]) {
        if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size())
            return;

        // dimensions = 4, CPU is NCHW, GPU is NHWC
        if (param.orank == 4) {
            int index = max_index(inTexture, gid.xy);
            outTexture.write(ftype4(index, 0.0, 0.0, 0.0), gid.xy, gid.z);
        }
        // dimensions < 4, CPU is NCHW, GPU treat as NHWC
        else {
            uint ix = gid.z * 4;
            uint iy = gid.x;
            int index_r = max_index(inTexture, uint2(ix, iy));
            int index_g = max_index(inTexture, uint2(ix + 1, iy));
            int index_b = max_index(inTexture, uint2(ix + 2, iy));
            int index_a = max_index(inTexture, uint2(ix + 3, iy));

            outTexture.write(ftype4(index_r, index_g, index_b, index_a), gid.xy, gid.z);
        }
    }
    ```

## 5. 添加 Argmax 单测

详细操作内容请见[新增单测](./add_unit_test)文档。
