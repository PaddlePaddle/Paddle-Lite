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

class ScaleComputeBuffer
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ScaleParam;

  std::string doc() const override { return "Scale using cl::Buffer, kFP16"; }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    scale_param_ = param_.get_mutable<param_t>();
    if (scale_param_->activation_type == "") {
      std::cout << "====111====" << std::endl;
      kernel_func_name_ = "scale_buffer";
    } else if (scale_param_->activation_type == "relu6") {
      kernel_func_name_ = "scale_relu6";
    } else {
      LOG(FATAL) << "Unsupported activation type: "
                 << scale_param_->activation_type;
    }
    if (scale_param_->fuse_scaleact) {
      kernel_func_name_ = "scaleacts";
    }
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/scale.cl", build_options_, time_stamp_);
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

      // compute global work size
      c_ = x_dims[1];
      w_ = x_dims[3];
      bh_ = x_dims[0] * x_dims[2];
      GetGlobalWorkSize();
    }
  }

  void GetGlobalWorkSize() {
    global_work_size_ = cl::NDRange{static_cast<cl::size_type>(bh_),
                                    static_cast<cl::size_type>(c_),
                                    static_cast<cl::size_type>((w_ + 7) / 8)};
  }

  void Run() override {
    auto x_dims = scale_param_->x->dims();
    auto* x_buf = GET_BUFFER_GPU(scale_param_->x);
    auto* out_buf =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? scale_param_->output->mutable_data<half_t, cl::Buffer>(
                  TARGET(kOpenCL))
            : scale_param_->output->mutable_data<float, cl::Buffer>(
                  TARGET(kOpenCL));
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
    status = kernel.setArg(0, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, scale);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, bias);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, alpha);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, static_cast<int>(x_dims[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(6, static_cast<int>(x_dims[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(7, static_cast<int>(x_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(8, static_cast<int>(x_dims[3]));
    CL_CHECK_FATAL(status);
    if (kernel_func_name_ == "scaleacts") {
      status = kernel.setArg(9, scale1);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(10, bias1);
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
  int c_;
  int w_;
  int bh_;
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
                     kNCHW,
                     paddle::lite::kernels::opencl::ScaleComputeBuffer,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
