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

class FcCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::FcParam;

  void PrepareForRun() override {
    fc_param_ = param_.get_mutable<param_t>();
    auto w_t = fc_param_->w;
    auto bias_t = fc_param_->bias;
    const bool enable_fp16 =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;

    auto device_name = CLRuntime::Global()->device().getInfo<CL_DEVICE_NAME>();
    if (device_name.find("Adreno") != std::string::npos) {
      is_adreno_ = true;
    }
    // TODO(sprouteer): optimize mali later
    is_adreno_ = true;

    if (fc_param_->activation_type == "relu") {
      build_options_ += " -DRELU";
    } else if (fc_param_->activation_type == "relu6") {
      build_options_ += " -DRELU6";
    } else if (fc_param_->activation_type == "prelu") {
      std::string prelu_mode = fc_param_->Prelu_mode;
      build_options_ += " -DPRELU";
      if (prelu_mode == "channel") {
        build_options_ += " -DPRELU_CH";
      } else if (prelu_mode == "element") {
        build_options_ += " -DPRELU_ELE";
      } else {
        build_options_ += " -DPRELU_ALL";
      }
      auto alpha_t = fc_param_->Prelu_alpha;
      alpha_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      auto alpha_gpu_data =
          alpha_gpu_t_->mutable_data(TARGET(kOpenCL), alpha_t->memory_size());
      TargetWrapperCL::MemcpySync(alpha_gpu_data,
                                  alpha_t->raw_data(),
                                  alpha_t->memory_size(),
                                  IoDirection::HtoD);
    }
    w_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    if (is_adreno_) {
      auto w_cpu_tensor = std::unique_ptr<Tensor>(new Tensor);
      CLImageConverterFolder w_converter;
      const DDim& w_image_dims =
          w_converter.InitImageDimInfoWith(fc_param_->w->dims());
      w_cpu_tensor->Resize({1, w_image_dims[0], w_image_dims[1], 4});
      auto* w_image_data = MUTABLE_DATA_CPU(w_cpu_tensor);
      auto* w_cpu = fc_param_->w->mutable_data<float>();
      w_converter.NCHWToImage(w_cpu, w_image_data, fc_param_->w->dims());
      MUTABLE_DATA_GPU(
          w_gpu_t_, w_image_dims[0], w_image_dims[1], w_image_data);
    } else {
      if (enable_fp16) {
        auto* w_cpu = w_t->data<float>();
        auto w_cpu_t = std::unique_ptr<Tensor>(new Tensor);
        w_cpu_t->Resize(w_t->dims());
        auto* w_buffer_data = MUTABLE_DATA_CPU(w_cpu_t.get());
        FloatArray2HalfArray(const_cast<float*>(w_cpu),
                             static_cast<half_t*>(w_buffer_data),
                             w_t->dims().production());
        auto w_gpu_data =
            w_gpu_t_->mutable_data(TARGET(kOpenCL), w_t->memory_size());
        TargetWrapperCL::MemcpySync(w_gpu_data,
                                    w_cpu_t->raw_data(),
                                    w_cpu_t->memory_size(),
                                    IoDirection::HtoD);
      } else {
        auto w_gpu_data =
            w_gpu_t_->mutable_data(TARGET(kOpenCL), w_t->memory_size());
        TargetWrapperCL::MemcpySync(
            w_gpu_data, w_t->raw_data(), w_t->memory_size(), IoDirection::HtoD);
      }
    }

    bias_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    if (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16) {
      auto* bias_cpu = bias_t->data<float>();
      auto bias_cpu_t = std::unique_ptr<Tensor>(new Tensor);
      bias_cpu_t->Resize(bias_t->dims());
      auto* bias_buffer_data = MUTABLE_DATA_CPU(bias_cpu_t.get());
      FloatArray2HalfArray(const_cast<float*>(bias_cpu),
                           static_cast<half_t*>(bias_buffer_data),
                           bias_t->dims().production());
      auto b_gpu_data =
          bias_gpu_t_->mutable_data(TARGET(kOpenCL), bias_t->memory_size());
      TargetWrapperCL::MemcpySync(b_gpu_data,
                                  bias_cpu_t->raw_data(),
                                  bias_cpu_t->memory_size(),
                                  IoDirection::HtoD);
    } else {
      auto b_gpu_data =
          bias_gpu_t_->mutable_data(TARGET(kOpenCL), bias_t->memory_size());
      TargetWrapperCL::MemcpySync(b_gpu_data,
                                  bias_t->raw_data(),
                                  bias_t->memory_size(),
                                  IoDirection::HtoD);
    }
  }

  void ReInitWhenNeeded() override {
    const auto x_dims = fc_param_->input->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute m,n,k
      const auto w_dims = fc_param_->w->dims();
      CHECK_GE(x_dims.size(), 2UL);
      CHECK_GE(w_dims.size(), 2UL);

      int in_num_col_dims = fc_param_->in_num_col_dims;
      std::string op_type = fc_param_->op_type;
      if (op_type == "matmul" || op_type == "matmul_v2") {
        in_num_col_dims = x_dims.size() - 1;
      }
      m_ = x_dims.Slice(0, in_num_col_dims).production();
      k_ = x_dims.Slice(in_num_col_dims, x_dims.size()).production();
      n_ = w_dims[1];
      CHECK_EQ(k_, static_cast<int>(w_dims[0]));

#ifdef LITE_WITH_LOG
      VLOG(4) << "x_dims:" << x_dims[0] << " " << x_dims[1] << " " << x_dims[2]
              << " " << x_dims[3];
      VLOG(4) << "w_dims:" << w_dims[0] << " " << w_dims[1] << " " << w_dims[2]
              << " " << w_dims[3];
      VLOG(4) << "m_: " << m_ << " n_: " << n_ << " k_: " << k_;
#endif

      // choose kernel
      // TODO(sprouteer): support mali later
      if (m_ == 1) {
        kernel_func_name_ = "adreno_gemv_1x4";
      } else {
        kernel_func_name_ = "adreno_gemm_4x4";
      }
#ifdef LITE_WITH_LOG
      VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
#endif

      if (fc_param_->activation_type == "relu") {
        build_options_ += "-DRELU";
      }

      auto& context = ctx_->As<OpenCLContext>();
      context.cl_context()->AddKernel(kernel_func_name_,
                                      "buffer/fc_kernel.cl",
                                      build_options_,
                                      time_stamp_);
      STL::stringstream kernel_key;
      kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
      kernel_ = context.cl_context()->GetKernel(kernel_key.str());

      // compute global work size
      GetGlobalWorkSize();
    }
  }

  void GetGlobalWorkSize() {
    if (m_ == 1) {  // gemv
      global_work_size_ = cl::NDRange{static_cast<size_t>((n_ + 3) / 4)};
    } else {  // gemm
      // TODO(sprouteer): Experience value, need add auto tune
      auto& context = ctx_->As<OpenCLContext>();
      size_t max_work_group_size = 0;
      kernel_.getWorkGroupInfo<size_t>(CLRuntime::Global()->device(),
                                       CL_KERNEL_WORK_GROUP_SIZE,
                                       &max_work_group_size);
      if (context.cl_context()->IsAppleM1() ||
          context.cl_context()->IsArmMali()) {
        local_work_size_ = cl::NullRange;
      } else {
        local_work_size_ = cl::NDRange(32, 32);
      }

      global_work_size_ = cl::NDRange{static_cast<size_t>((m_ + 3) / 4),
                                      static_cast<size_t>((n_ + 3) / 4)};
    }
  }

  void Run() override {
    auto* x_buf = GET_BUFFER_GPU(fc_param_->input);
    auto* bias_buf = GET_BUFFER_GPU(bias_gpu_t_);
    const cl::Buffer* alpha_buf = nullptr;
    if (fc_param_->activation_type == "prelu") {
      alpha_buf = alpha_gpu_t_->data<float, cl::Buffer>();
    }
    auto* out_buf = MUTABLE_BUFFER_GPU(fc_param_->output);

    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_buf);
    CL_CHECK_FATAL(status);
    if (is_adreno_) {
      auto* w_img = GET_DATA_GPU(w_gpu_t_);
      status = kernel.setArg(1, *w_img);
      CL_CHECK_FATAL(status);
    } else {
      auto* w_buf = GET_BUFFER_GPU(w_gpu_t_);
      status = kernel.setArg(1, *w_buf);
      CL_CHECK_FATAL(status);
    }
    status = kernel.setArg(2, *bias_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, static_cast<const int>(m_));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, static_cast<const int>(n_));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(6, static_cast<const int>(k_));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(7, *alpha_buf);
    CL_CHECK_FATAL(status);

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  local_work_size_,
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
  param_t* fc_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  bool is_adreno_{false};
  std::unique_ptr<Tensor> w_gpu_t_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_t_{nullptr};
  std::unique_ptr<Tensor> alpha_gpu_t_{nullptr};

  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_;
  cl::Kernel kernel_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    fc, kOpenCL, kFP16, kNCHW, paddle::lite::kernels::opencl::FcCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fc, kOpenCL, kFP16, kNCHW, paddle::lite::kernels::opencl::FcCompute, pc)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
