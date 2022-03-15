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

#include <cmath>
#include <memory>
#include <string>
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/host/elementwise_op_func.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/log/logging.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"


namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {
class LogicalImageCompute : public KernelLite<TARGET(kOpenCL),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::LogicalParam;

  void PrepareForRun() override {
    logical_param_ = param_.get_mutable<param_t>();
    auto* x = logical_param_->X;
    auto* y = logical_param_->Y;
    x_dims_ = logical_param_->X->dims();
    y_dims_ = logical_param_->Y->dims();
    auto& context = ctx_->As<OpenCLContext>();
    act_type_ = static_cast<int>(logical_param_->logical_type);
    switch (act_type_){
      case 0:
        kernel_func_name_="xor";
        break;
      default:
        LOG(FATAL) << "This act type:" << act_type_ << " doesn't support.";
        return;

    }
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/logical_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {

    auto x_dims = logical_param_->X->dims();
    auto out_dim =logical_param_->Out->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute image shape
      paddle::lite::CLImageConverterDefault default_convertor;
      x_img_shape_ = default_convertor.InitImageDimInfoWith(
          x_dims);  // w, h
      out_img_shape_ = default_convertor.InitImageDimInfoWith(
          out_dim);  // w, h

      // compute global work size
      GetGlobalWorkSize();
    }
  }
  void GetGlobalWorkSize() {
    global_work_size_ =
        cl::NDRange{static_cast<cl::size_type>(x_img_shape_[0]),
                    static_cast<cl::size_type>(x_img_shape_[1])};
  }

  void Run() override {
    auto* x_img = GET_DATA_GPU(logical_param_->X);
    auto* y_img = GET_BUFFER_GPU(logical_param_->Y);
    auto* out_img = MUTABLE_DATA_GPU(
        logical_param_->Out, out_img_shape_[0], out_img_shape_[1], nullptr);
    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *y_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);

#ifdef LITE_WITH_LOG
    const auto& x_dims = logical_param_->X->dims();
    const auto& y_dims = logical_param_->Out->dims();
    VLOG(4) << TargetToStr(logical_param_->X->target());
    VLOG(4) << TargetToStr(logical_param_->Out->target());
    VLOG(4) << "x_img_shape_(w,h):" << x_img_shape_[0] << " "
            << x_img_shape_[1];
    VLOG(4) << "kernel func name:" << kernel_func_name_;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }
#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch){
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;
  }
#endif
  private:
  param_t* logical_param_{nullptr};
  DDim last_x_dims_;
  std::string time_stamp_{GetTimeStamp()};
  std::string kernel_func_name_{};
  cl::Kernel kernel_;
  int act_type_{0};
  std::string build_options_{""};
  bool first_epoch_for_reinit_{true};
  DDimLite x_dims_{};
  DDimLite y_dims_{};
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1)};
  DDim x_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
};
}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
/*REGISTER_LITE_KERNEL(
    logical_not,
    kOpenCL,
    kBool,
    kImageDefault,
    paddle::lite::kernels::opencl::LogicalImageCompute,
    def)
.BindInput("X",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kBool),
    DATALAYOUT(kImageDefault))})
.BindOutput("Out",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kBool),
    DATALAYOUT(kImageDefault))})
.Finalize();

REGISTER_LITE_KERNEL(
    logical_or,
    kOpenCL,
    kBool,
    kImageDefault,
    paddle::lite::kernels::opencl::LogicalImageCompute,
    def)
.BindInput("X",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kBool),
    DATALAYOUT(kImageDefault))})
.BindInput("Y",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kBool),
    DATALAYOUT(kImageDefault))})
.BindOutput("Out",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kBool),
    DATALAYOUT(kImageDefault))})
.Finalize();

REGISTER_LITE_KERNEL(
    logical_and,
    kOpenCL,
    kBool,
    kImageDefault,
    paddle::lite::kernels::opencl::LogicalImageCompute,
    def)
.BindInput("X",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kBool),
    DATALAYOUT(kImageDefault))})
.BindInput("Y",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kBool),
    DATALAYOUT(kImageDefault))})
.BindOutput("Out",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kBool),
    DATALAYOUT(kImageDefault))})
.Finalize();
*/
REGISTER_LITE_KERNEL(
    logical_xor,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::LogicalImageCompute,
    def)
.BindInput("X",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kFP16),
    DATALAYOUT(kImageDefault))})
.BindInput("Y",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kFP16),
    DATALAYOUT(kImageDefault))})
.BindOutput("Out",
{LiteType::GetTensorTy(TARGET(kOpenCL),
    PRECISION(kFP16),
    DATALAYOUT(kImageDefault))})
.Finalize();

