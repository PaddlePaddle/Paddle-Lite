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

#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ScaleComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ScaleParam;

  std::string doc() const override { return "Scale using cl::Image2D, kFP16"; }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    scale_param_ = param_.get_mutable<param_t>();
    if (scale_param_->activation_type == "") {
      kernel_func_name_ = "scale";
    } else if (scale_param_->activation_type == "relu6") {
      kernel_func_name_ = "scale_relu6";
    } else {
      LOG(FATAL) << "Unsupported activation type: "
                 << scale_param_->activation_type;
    }
    if (scale_param_->fuse_scaleact) {
      kernel_func_name_ = "scaleacts";
    }
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/scale_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    scale_param_ = param_.get_mutable<param_t>();
    auto x_dims = scale_param_->x->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute image shape
      paddle::lite::CLImageConverterDefault default_convertor;
      out_img_shape_ =
          default_convertor.InitImageDimInfoWith(scale_param_->output->dims());

      // compute global work size
      GetGlobalWorkSize();
    }
  }

  void GetGlobalWorkSize() {
    global_work_size_ =
        cl::NDRange{static_cast<cl::size_type>(out_img_shape_[0]),
                    static_cast<cl::size_type>(out_img_shape_[1])};
  }

  void Run() override {
    auto* x_img = GET_DATA_GPU(scale_param_->x);
    auto* out_img = MUTABLE_DATA_GPU(
        scale_param_->output, out_img_shape_[0], out_img_shape_[1], nullptr);
    const float scale = scale_param_->scale;
    float bias = scale_param_->bias;
    if (!scale_param_->bias_after_scale) {
      bias *= scale;
    }
    const float alpha = scale_param_->alpha;
    const float scale1 = scale_param_->scale1;
    const float bias1 = scale_param_->bias1;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, scale);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, bias);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, alpha);
    CL_CHECK_FATAL(status);
    if (kernel_func_name_ == "scaleacts") {
      status = kernel.setArg(5, scale1);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(6, bias1);
      CL_CHECK_FATAL(status);
    }

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
  std::string kernel_func_name_{"scale"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};

  param_t* scale_param_{nullptr};
  cl::Kernel kernel_;
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(scale,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ScaleComputeImage2D,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
