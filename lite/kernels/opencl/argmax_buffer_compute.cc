// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ArgmaxComputeBuffer
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ArgmaxParam;

  std::string doc() const override { return "Argmax using cl::Buffer, kFP16"; }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    argmax_param_ = param_.get_mutable<param_t>();
    auto& x_dims = argmax_param_->X->dims();
    axis_ = argmax_param_->Axis;

    if (axis_ < 0) axis_ += x_dims.size();
    switch (argmax_param_->dtype) {
      // default indices type: int64_t
      case -1: {
        kernel_func_name_ = "argmax_out_int64";
        break;
      }
      // static_cast<int>(lite::core::FluidType::INT32) == 2
      case 2: {
        kernel_func_name_ = "argmax_out_int32";
        break;
      }
      // static_cast<int>(lite::core::FluidType::INT64) == 3
      case 3: {
        kernel_func_name_ = "argmax_out_int64";
        break;
      }
      default: {
        LOG(FATAL) << "Attribute `dtype` in arg_max op must be 2 or 3, which "
                      "indicates that indices dtype must be int32 or int64, "
                      "default dtype is int64.";
        break;
      }
    }
    create_build_options();

    context.cl_context()->AddKernel(kernel_func_name_,
                                    "buffer/argmax_buffer.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    argmax_param_ = param_.get_mutable<param_t>();
    auto& x_dims = argmax_param_->X->dims();

    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      outer_size_ = x_dims.count(0, axis_);
      axis_size_ = x_dims[axis_];
      inner_size_ = x_dims.count(axis_ + 1, x_dims.size());
      gws_ = cl::NDRange{static_cast<cl::size_type>(inner_size_),
                         static_cast<cl::size_type>(outer_size_)};
    }
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto* x_buf = GET_BUFFER_GPU(argmax_param_->X);
    cl::Buffer* out_buf;
    switch (argmax_param_->dtype) {
      // default indices type: int64_t
      case -1: {
        out_buf = argmax_param_->Out->mutable_data<int64_t, cl::Buffer>(
            TARGET(kOpenCL));
        break;
      }
      // static_cast<int>(lite::core::FluidType::INT32) == 2
      case 2: {
        out_buf = argmax_param_->Out->mutable_data<int32_t, cl::Buffer>(
            TARGET(kOpenCL));
        break;
      }
      // static_cast<int>(lite::core::FluidType::INT64) == 3
      case 3: {
        out_buf = argmax_param_->Out->mutable_data<int64_t, cl::Buffer>(
            TARGET(kOpenCL));
        break;
      }
      default: {
        LOG(FATAL) << "Attribute `dtype` in arg_max op must be 2 or 3, which "
                      "indicates that indices dtype must be int32 or int64, "
                      "default dtype is int64.";
        break;
      }
    }
    cl_int status;
    int arg_idx = 0;
    status = kernel_.setArg(arg_idx++, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, outer_size_);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, axis_size_);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, inner_size_);
    CL_CHECK_FATAL(status);
    status = EnqueueNDRangeKernel(
        context, kernel_, cl::NullRange, gws_, cl::NullRange, nullptr, event_);
    CL_CHECK_FATAL(status);
  }

  void create_build_options() {
    const bool fp16_support =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
    std::string init_max = " -DDATAINIT=-FLT_MAX ";
    std::string flag_type =
        fp16_support ? " -DFLAG_TYPE8=short8 " : " -DFLAG_TYPE8=int8 ";
    build_options_ = init_max + flag_type;
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->global_work_size = ch->NDRangeToStr(gws_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  param_t* argmax_param_{nullptr};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  int axis_{-1};
  int outer_size_{1};
  int axis_size_{1};
  int inner_size_{1};
  std::string kernel_func_name_{};
  std::string build_options_{};
  std::string time_stamp_{GetTimeStamp()};
  cl::Kernel kernel_;
  cl::NDRange gws_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(arg_max,
                     kOpenCL,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::opencl::ArgmaxComputeBuffer,
                     fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
