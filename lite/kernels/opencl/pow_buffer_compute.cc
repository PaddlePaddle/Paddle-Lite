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

#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class PowComputeBuffer
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::PowParam;

  std::string doc() const override { return "Pow using cl::Buffer, kFP16"; }

  void PrepareForRun() override {
    pow_param_ = param_.get_mutable<param_t>();
    auto x_dims = pow_param_->X->dims();
    bool fp16_support =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
    auto type = pow_param_->X->target();
    if (type == TargetType::kHost || type == TargetType::kARM) {
      x_persistable_ = true;
    }
    if (x_persistable_) {
      if (fp16_support) {
        pow_x_buf_t_ = std::unique_ptr<Tensor>(new Tensor);
        const auto pow_x_dims = pow_param_->X->dims();
        auto* pow_x_cpu = pow_param_->X->data<float>();
        auto pow_x_cpu_t = std::unique_ptr<Tensor>(new Tensor);
        pow_x_cpu_t->Resize(pow_x_dims);
        auto* pow_x_buffer_data = MUTABLE_DATA_CPU(pow_x_cpu_t.get());
        FloatArray2HalfArray(const_cast<float*>(pow_x_cpu),
                             static_cast<half_t*>(pow_x_buffer_data),
                             pow_x_dims.production());
        auto* pow_x_gpu_data = pow_x_buf_t_->mutable_data(
            TARGET(kOpenCL), pow_x_cpu_t->memory_size());
        TargetWrapperCL::MemcpySync(pow_x_gpu_data,
                                    pow_x_cpu_t->raw_data(),
                                    pow_x_cpu_t->memory_size(),
                                    IoDirection::HtoD);
      } else {
        pow_x_buf_t_ = std::unique_ptr<Tensor>(new Tensor);
        auto pow_x_gpu_data = pow_x_buf_t_->mutable_data(
            TARGET(kOpenCL), pow_param_->X->memory_size());
        TargetWrapperCL::MemcpySync(pow_x_gpu_data,
                                    pow_param_->X->raw_data(),
                                    pow_param_->X->memory_size(),
                                    IoDirection::HtoD);
      }
    }

    kernel_func_name_ = "pow_buffer";
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/pow_kernel.cl", build_options_, time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

  void ReInitWhenNeeded() override {
    pow_param_ = param_.get_mutable<param_t>();
    auto x_dims = pow_param_->X->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;
      size_t count = x_dims.production();
      global_work_size_ = cl::NDRange{count};
    }
  }

  void Run() override {
    auto x_dims = pow_param_->X->dims();
    auto* out_buf =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? pow_param_->Out->mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL))
            : pow_param_->Out->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    float scale = 1.0;
    float shift = 0.0;
    float power = pow_param_->factor;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto& kernel = kernel_;
    cl_int status;
    if (x_persistable_) {
      auto* x_buf = GET_BUFFER_GPU(pow_x_buf_t_);
      status = kernel.setArg(0, *x_buf);
      CL_CHECK_FATAL(status);
    } else {
      auto* x_buf = GET_BUFFER_GPU(pow_param_->X);
      status = kernel.setArg(0, *x_buf);
      CL_CHECK_FATAL(status);
    }
    status = kernel.setArg(1, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, static_cast<float>(scale));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, static_cast<float>(shift));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, static_cast<float>(power));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, static_cast<int>(x_dims.production()));
    CL_CHECK_FATAL(status);

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
    ch->global_work_size = ch->NDRangeToStr(global_work_size_);
    ch->local_work_size = ch->NDRangeToStr(local_work_size_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  std::string kernel_func_name_{""};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  bool x_persistable_{false};
  std::unique_ptr<Tensor> pow_x_buf_t_;
  param_t* pow_param_{nullptr};
  cl::Kernel kernel_;
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  bool onexone_flag_{false};
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_{cl::NullRange};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pow,
                     kOpenCL,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::opencl::PowComputeBuffer,
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
