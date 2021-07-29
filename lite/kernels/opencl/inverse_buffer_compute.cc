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

#include <iostream>
#include "lite/core/op_registry.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/kernels/opencl/image_helper.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class InverseBufferCompute : public KernelLite<TARGET(kOpenCL),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kNCHW)> {
public:
  using param_t = operators::InverseParam;

  std::string doc() const override { return "Inverse using cl::Buffer, kFloat"; }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    kernel_func_names_.push_back("initialize_out_as_identity");
    kernel_func_names_.push_back("partial_gauss_elimination");
    kernel_func_paths_.push_back("buffer/inverse_kernel.cl");
    for (size_t i = 0; i < kernel_func_names_.size(); i++) {
      context.cl_context()->AddKernel(kernel_func_names_[i],
                                      kernel_func_paths_[0],
                                      build_options_,
                                      time_stamp_);
    }

    STL::stringstream kernel_key;
    kernel_key.str("");
    kernel_key << kernel_func_names_[0] << build_options_ << time_stamp_;
    kernel_initialize_out_identity = context.cl_context()->GetKernel(kernel_key.str());
    kernel_key.str("");
    kernel_key << kernel_func_names_[1] << build_options_ << time_stamp_;
    kernel_partial_gauss_elimination = context.cl_context()->GetKernel(kernel_key.str());

    inverse_param_ = param_.get_mutable<param_t>();
    const auto x_dims = inverse_param_->Input->dims();
    x_n_ = x_dims[0];
    x_c_ = x_dims[1];
    x_h_ = x_dims[2];
    x_w_ = x_dims[3];
  }

  void ReInitWhenNeeded() override {
    const auto x_dims = inverse_param_->Input->dims();
    const auto out_dims = inverse_param_->Output->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;
    
      x_n_ = x_dims[0];
      x_c_ = x_dims[1];
      x_h_ = x_dims[2];
      x_w_ = x_dims[3];

      GetGlobalWorkSize();
    }
  }

  void GetGlobalWorkSize() {
    global_work_size_initial_out = 
        cl::NDRange{static_cast<cl::size_type>(x_h_),
                    static_cast<cl::size_type>(x_w_),
                    static_cast<cl::size_type>(x_c_ * x_n_)};
    global_work_size_gauss =
        cl::NDRange{static_cast<cl::size_type>(1),
                    static_cast<cl::size_type>(x_w_),
                    static_cast<cl::size_type>(x_c_ * x_n_)};

#ifdef LITE_WITH_LOG
    VLOG(4) << "global_work_size:" << x_h_ << " " << x_w_ << " " << x_c_ * x_n_;
#endif
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    cl_int status;

    int length = inverse_param_->Input->dims().production();
    auto* x_buf = inverse_param_->Input->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    auto* out_buf = inverse_param_->Output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

#ifdef LITE_WITH_LOG
    VLOG(4) << TargetToStr(inverse_param_->Input->target());
    VLOG(4) << TargetToStr(inverse_param_->Output->target());
#endif

    // kernel 1
    status = kernel_initialize_out_identity.setArg(0, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel_initialize_out_identity.setArg(1, (const int)x_h_);
    CL_CHECK_FATAL(status);
    status = kernel_initialize_out_identity.setArg(2, (const int)x_w_);
    CL_CHECK_FATAL(status);
    status = kernel_initialize_out_identity.setArg(3, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel_initialize_out_identity.setArg(4, length);
    CL_CHECK_FATAL(status);
    status = EnqueueNDRangeKernel(context,
                                  kernel_initialize_out_identity,
                                  cl::NullRange,
                                  global_work_size_initial_out,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
    
    // kernel 2
    local_work_size_ = cl::NDRange(1, x_w_, 1);
    status = kernel_partial_gauss_elimination.setArg(0, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel_partial_gauss_elimination.setArg(1, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel_partial_gauss_elimination.setArg(2, (const int)x_h_);
    CL_CHECK_FATAL(status);
    status = kernel_partial_gauss_elimination.setArg(3, (const int)x_w_);
    CL_CHECK_FATAL(status);
    status = EnqueueNDRangeKernel(context,
                                  kernel_partial_gauss_elimination,
                                  cl::NullRange,
                                  global_work_size_gauss,
                                  local_work_size_,//cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_names_[0];
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

protected:
  std::vector<std::string> kernel_func_names_{};
  std::vector<std::string> kernel_func_paths_{};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::string time_stamp_{GetTimeStamp()};

  int x_n_{-1};
  int x_c_{-1};
  int x_h_{-1};
  int x_w_{-1};

  param_t* inverse_param_{nullptr};
  cl::Kernel kernel_initialize_out_identity;
  cl::Kernel kernel_partial_gauss_elimination;
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;

  cl::NDRange global_work_size_initial_out = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange global_work_size_gauss = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange local_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;

REGISTER_LITE_KERNEL(inverse,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::InverseBufferCompute,
                     def)
    .BindInput("Input",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kFloat))})
    .Finalize();
