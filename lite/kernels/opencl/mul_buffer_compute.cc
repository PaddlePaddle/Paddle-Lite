// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

class MulCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::MulParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "buffer/mat_mul_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    const auto& param = *param_.get_mutable<param_t>();
    const auto* x_data = param.x->data<float>();
    const auto* y_data = param.y->data<float>();
    auto* o_data = param.output->mutable_data<float>();

    m_ = static_cast<int>(
        param.x->dims().Slice(0, param.x_num_col_dims).production());
    const int x_w = static_cast<int>(
        param.x->dims()
            .Slice(param.x_num_col_dims, param.x->dims().size())
            .production());
    int y_h = static_cast<int>(
        param.y->dims().Slice(0, param.y_num_col_dims).production());
    n_ = static_cast<int>(
        param.y->dims()
            .Slice(param.y_num_col_dims, param.y->dims().size())
            .production());

    CHECK_EQ(x_w, y_h) << "x_w must be equal with y_h";
    k_ = x_w;
    VLOG(4) << "m: " << m_ << " n_: " << n_ << " k_: " << k_ << " y_h: " << y_h
            << " x_w: " << x_w;
  }

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* x_buf = param.x->data<float, cl::Buffer>();
    auto* y_buf = param.y->data<float, cl::Buffer>();
    auto* out_buf =
        param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    cl_int status;
    int arg_idx = 0;
    status = kernel.setArg(arg_idx, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, m_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, n_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, k_);
    CL_CHECK_FATAL(status);

    auto global_work_size = cl::NDRange{static_cast<size_t>((m_ + 3) / 4),
                                        static_cast<size_t>((n_ + 3) / 4)};

    status = EnqueueNDRangeKernel(context,
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
  int m_, n_, k_;
  std::string kernel_func_name_{"mat_mul"};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    mul, kOpenCL, kFloat, kNCHW, paddle::lite::kernels::opencl::MulCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
