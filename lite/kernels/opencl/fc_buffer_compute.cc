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
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class FcCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::FcParam;

  void PrepareForRun() override {
    const auto& param = *param_.get_mutable<param_t>();
    const auto x_dims = param.input->dims();
    const auto w_dims = param.w->dims();

    CHECK_GE(x_dims.size(), 2UL);
    CHECK_GE(w_dims.size(), 2UL);
    CHECK_EQ(param.output->dims().size(), 2UL);

    m_ = x_dims.Slice(0, param.in_num_col_dims).production();
    k_ = x_dims.Slice(param.in_num_col_dims, x_dims.size()).production();
    n_ = w_dims[1];
    CHECK_EQ(k_, static_cast<int>(w_dims[0]));
    VLOG(4) << "x_dims:" << x_dims[0] << " " << x_dims[1] << " " << x_dims[2]
            << " " << x_dims[3];
    VLOG(4) << "w_dims:" << w_dims[0] << " " << w_dims[1] << " " << w_dims[2]
            << " " << w_dims[3];
    VLOG(4) << "m_: " << m_ << " n_: " << n_ << " k_: " << k_;

    if (m_ == 1) {  // gemv
      kernel_func_name_ = "fc_gemv_1x4";
      global_work_size_ = cl::NDRange{static_cast<size_t>((n_ + 3) / 4)};
    } else {  // gemm
      kernel_func_name_ = "fc_gemm_4x4";
      global_work_size_ = cl::NDRange{static_cast<size_t>((m_ + 3) / 4),
                                      static_cast<size_t>((n_ + 3) / 4)};
    }
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/fc_kernel.cl", build_options_);
  }

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* x_buf = param.input->data<float, cl::Buffer>();
    auto* w_buf = param.w->data<float, cl::Buffer>();
    auto* bias_buf = param.bias->data<float, cl::Buffer>();
    auto* out_buf =
        param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    cl_int status;
    int arg_idx = 0;
    status = kernel.setArg(arg_idx, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *w_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *bias_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(m_));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(n_));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(k_));
    CL_CHECK_FATAL(status);

    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size_,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    context.cl_wait_list()->emplace(out_buf, event_);
  }

 private:
  int m_, n_, k_;
  std::string kernel_func_name_{};
  std::string build_options_{"-DCL_DTYPE=float"};
  cl::NDRange global_work_size_;
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    fc, kOpenCL, kFloat, kNCHW, paddle::lite::kernels::opencl::FcCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
