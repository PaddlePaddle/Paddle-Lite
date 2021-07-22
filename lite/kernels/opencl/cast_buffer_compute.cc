// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class CastCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::CastParam;

  std::string doc() const override { return "Cast using cl::Buffer, kFloat"; }
  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    auto& param = *param_.get_mutable<param_t>();
    int in_type = param.in_dtype;
    int out_type = param.out_dtype;

    // cast int32 to float and its inverse
    if (in_type == 2 && out_type == 5) {
      kernel_func_name_ = std::string("cast_int_to_float");
      std::string build_options_{"-DCL_DTYPE_float"};
    } else if (in_type == 5 && out_type == 2) {
      kernel_func_name_ = std::string("cast_float_to_int");
      std::string build_options_{"-DCL_DTYPE_float"};
    } else {
      LOG(FATAL) << "Attribute `dtype` in cast op must be 2 or 5, which "
                    "indicates that in / ouput dtype must be int32 or float, "
                    "only suport int32 to float and float to int32 cast"
                    "default dtype is int32.";
    }

    VLOG(4) << "kernel function: " << kernel_func_name_
            << "\tinput_type: " << in_type << "\toutput_type: " << out_type;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "buffer/cast_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    size_t count = x_dims.production();

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    // data type
    int in_type = param.in_dtype;
    int out_type = param.out_dtype;

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;

    auto kernel = context.cl_context()->GetKernel(kernel_key.str());
    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Out->target());

    int arg_idx = 0;
    if (in_type == 2 && out_type == 5) {
      auto* x_buf = param.X->data<int, cl::Buffer>();
      auto* out_buf =
          param.Out->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

      cl_int status = kernel.setArg(arg_idx, *x_buf);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, (const int)count);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *out_buf);
      CL_CHECK_FATAL(status);

    } else if (in_type == 5 && out_type == 2) {
      auto* x_buf = param.X->data<float, cl::Buffer>();
      auto* out_buf = param.Out->mutable_data<int, cl::Buffer>(TARGET(kOpenCL));
      cl_int status = kernel.setArg(arg_idx, *x_buf);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, (const int)count);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *out_buf);
      CL_CHECK_FATAL(status);
    } else {
      LOG(FATAL) << "Attribute `dtype` in cast op must be 2 or 5, which "
                    "indicates that indices dtype must be int32 or float, "
                    "only suport int32 to float and float to int32 cast"
                    "default dtype is int32.";
    }

    auto global_work_size = cl::NDRange{count};
    cl_int status = EnqueueNDRangeKernel(context,
                                         kernel,
                                         cl::NullRange,
                                         global_work_size,
                                         cl::NullRange,
                                         nullptr,
                                         event_);
    CL_CHECK_FATAL(status);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  std::string kernel_func_name_{"cast_int_to_float"};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// Cast float to int32
REGISTER_LITE_KERNEL(cast,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::CastCompute,
                     float_to_int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kInt32))})
    .Finalize();

// Cast int32 to float
REGISTER_LITE_KERNEL(cast,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::CastCompute,
                     int32_to_float)
    .BindOutput("X",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kInt32))})
    .BindInput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
