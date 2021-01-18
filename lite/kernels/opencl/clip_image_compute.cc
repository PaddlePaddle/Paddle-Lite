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

#include <limits>

#include "lite/backends/opencl/cl_half.h"
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

class ClipComputeImageDefault : public KernelLite<TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ClipParam;

  std::string doc() const override {
    return "Clip using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    clip_param_ = param_.get_mutable<param_t>();
    lite::Tensor* min_tensor = clip_param_->min_tensor;
    lite::Tensor* max_tensor = clip_param_->max_tensor;
    min_ = clip_param_->min;
    max_ = clip_param_->max;

    if (min_tensor != nullptr) {
      min_ = min_tensor->data<float>()[0];
    }
    if (max_tensor != nullptr) {
      max_ = max_tensor->data<float>()[0];
    }

#ifdef LITE_WITH_LOG
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/clip_kernel.cl", build_options_, time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    auto x_dims = clip_param_->x->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute image shape
      paddle::lite::CLImageConverterDefault default_convertor;
      x_img_shape_ = default_convertor.InitImageDimInfoWith(
          clip_param_->x->dims());  // w, h
      out_img_shape_ = default_convertor.InitImageDimInfoWith(
          clip_param_->out->dims());  // w, h

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
    auto* x_img = GET_DATA_GPU(clip_param_->x);
    auto* out_img = MUTABLE_DATA_GPU(
        clip_param_->out, out_img_shape_[0], out_img_shape_[1], nullptr);
    auto kernel = kernel_;

    cl_int status;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, min_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, max_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, *out_img);
    CL_CHECK_FATAL(status);

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
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  param_t* clip_param_{nullptr};
  DDim x_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim last_x_dims_;

  std::string kernel_func_name_{"clip"};
  float max_{std::numeric_limits<float>::max()};  // FLT_MAX
  int use_max_{1};
  float min_{std::numeric_limits<float>::min()};  // FLT_MIN
  int use_min_{1};

  cl::Kernel kernel_;
  bool first_epoch_for_reinit_{true};
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
REGISTER_LITE_KERNEL(clip,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ClipComputeImageDefault,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Min", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Max", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindPaddleOpVersion("clip", 1)
    .Finalize();
