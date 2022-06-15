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
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::FcParam;

  void TransW(float* src, float* dst, int n, int k) {
    if (src == nullptr || dst == nullptr) return;
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        dst[i * n + j] = src[j * k + i];
        // std::cout << "i:" << i << "; j:" << j << "->" << dst[i * n + j] <<
        // std::endl;
      }
    }
  }

  void PrepareForRun() override {
    fc_param_ = param_.get_mutable<param_t>();
    auto w_t = fc_param_->w;
    auto bias_t = fc_param_->bias;
    auto device_name = CLRuntime::Global()->device().getInfo<CL_DEVICE_NAME>();
    if (device_name.find("Adreno") == std::string::npos) {
      is_adreno_ = false;
    }
    is_adreno_ = true;
    prepare_trans_w_ = false;
    std::cout << "is_adreno_" << is_adreno_ << std::endl;
    std::cout << "prepare_trans_w_" << prepare_trans_w_ << std::endl;
    if (fc_param_->activation_type == "prelu") {
      std::string prelu_mode = fc_param_->Prelu_mode;
      build_options_ += " -DPRELU";
      if (prelu_mode == "all") {
        build_options_ += " -DPRELU_ONE";
      } else {
        build_options_ += " -DPRELU_MORE";
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
    if (double_buffer_) {
      const auto w_dims = w_t->dims();
      auto* w_cpu = fc_param_->w->mutable_data<float>();
      auto w_cpu_t = std::unique_ptr<Tensor>(new Tensor);
      w_cpu_t->Resize(w_dims);
      auto* w_buffer_data = MUTABLE_DATA_CPU(w_cpu_t.get());
      bool fp16_support =
          CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
      FloatArray2HalfArray(static_cast<float*>(w_cpu),
                           static_cast<half_t*>(w_buffer_data),
                           w_dims.production());
      auto* w_gpu_data =
          w_gpu_t_->mutable_data(TARGET(kOpenCL), w_cpu_t->memory_size());
      TargetWrapperCL::MemcpySync(w_gpu_data,
                                  w_cpu_t->raw_data(),
                                  w_cpu_t->memory_size(),
                                  IoDirection::HtoD);
    } else if (prepare_trans_w_ && fc_param_->trans_weights == true) {
      const auto w_dims = w_t->dims();
      DDim w_trans_dims =
          DDim(std::vector<DDim::value_type>{w_dims[1], w_dims[0]});
      std::cout << "w_trans_dims: " << w_trans_dims[0] << "; "
                << w_trans_dims[1] << std::endl;
      auto w_cpu_tensor = std::unique_ptr<Tensor>(new Tensor);
      CLImageConverterFolder w_converter;
      const DDim& w_image_dims = w_converter.InitImageDimInfoWith(w_trans_dims);
      w_cpu_tensor->Resize({1, w_image_dims[0], w_image_dims[1], 4});
      auto* w_image_data = MUTABLE_DATA_CPU(w_cpu_tensor);
      auto* w_cpu = fc_param_->w->mutable_data<float>();
      std::vector<float> w_trans_cpu(w_trans_dims.production());
      TransW(w_cpu, w_trans_cpu.data(), w_dims[0], w_dims[1]);
      w_converter.NCHWToImage(w_trans_cpu.data(), w_image_data, w_trans_dims);
      MUTABLE_DATA_GPU(
          w_gpu_t_, w_image_dims[0], w_image_dims[1], w_image_data);
    } else {
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
    }

    bias_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    auto b_gpu_data =
        bias_gpu_t_->mutable_data(TARGET(kOpenCL), bias_t->memory_size());
    TargetWrapperCL::MemcpySync(b_gpu_data,
                                bias_t->raw_data(),
                                bias_t->memory_size(),
                                IoDirection::HtoD);
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
      // CHECK_EQ(fc_param_->output->dims().size(), 2UL);

      int in_num_col_dims = fc_param_->in_num_col_dims;
      std::string op_type = fc_param_->op_type;
      op_type = "matmul_v2";
      // std::cout << "op_type" << op_type << std::endl;
      if (op_type == "matmul" || op_type == "matmul_v2") {
        in_num_col_dims = x_dims.size() - 1;
        // std::cout << "in_num_col_dims" << in_num_col_dims << std::endl;
      }
      m_ = x_dims.Slice(0, in_num_col_dims).production();
      k_ = x_dims.Slice(in_num_col_dims, x_dims.size()).production();
      // n_ = w_dims[1];
      if (fc_param_->trans_weights == true) {
        CHECK_EQ(k_, static_cast<int>(w_dims[1]));
        n_ = w_dims[0];
      } else {
        n_ = w_dims[1];
        CHECK_EQ(k_, static_cast<int>(w_dims[0]));
      }

#ifdef LITE_WITH_LOG
      VLOG(4) << "x_dims:" << x_dims[0] << " " << x_dims[1] << " " << x_dims[2]
              << " " << x_dims[3];
      VLOG(4) << "w_dims:" << w_dims[0] << " " << w_dims[1] << " " << w_dims[2]
              << " " << w_dims[3];
      VLOG(4) << "m_: " << m_ << " n_: " << n_ << " k_: " << k_;
#endif

      // choose kernel
      auto device_name =
          CLRuntime::Global()->device().getInfo<CL_DEVICE_NAME>();
      if (double_buffer_) {
        if (m_ == 1) {
          kernel_func_name_ = "fc_gemv_1x4";
        } else {
          kernel_func_name_ = "mali_gemm_trans_4x4";
          if (fc_param_->trans_weights == true)
            kernel_func_name_ = "mali_gemm_trans_4x4";
        }
      } else {
        if (m_ == 1) {
          kernel_func_name_ = "adreno_gemv_1x4";
          if (fc_param_->trans_weights == true && prepare_trans_w_ == false)
            kernel_func_name_ = "adreno_gemv_trans_1x4";
        } else {
          kernel_func_name_ = "adreno_gemm_4x4";
          if (fc_param_->trans_weights == true && prepare_trans_w_ == false)
            kernel_func_name_ = "adreno_gemm_trans_4x4";
        }
      }
      // if (is_adreno_) {
      //   if (m_ == 1) {
      //     kernel_func_name_ = "adreno_gemv_1x4";
      //     if (fc_param_->trans_weights == true) kernel_func_name_ =
      //     "adreno_gemv_trans_1x4";
      //   } else {
      //     kernel_func_name_ = "adreno_gemm_4x4";
      //     if (fc_param_->trans_weights == true) kernel_func_name_ =
      //     "adreno_gemm_trans_4x4";
      //   }
      // } else {
      //   if (m_ == 1) {
      //     kernel_func_name_ = "fc_gemv_1x4";
      //   } else {
      //     kernel_func_name_ = "mali_gemm_trans_4x4";
      //     if (fc_param_->trans_weights == true) kernel_func_name_ =
      //     "mali_gemm_trans_4x4";
      //   }
      // }
      std::cout << "======= " << kernel_func_name_ << std::endl;
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
      // local_work_size_ = cl::NDRange(32, 4, 16);
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
    auto* out_buf =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? fc_param_->output->mutable_data<half_t, cl::Buffer>(
                  TARGET(kOpenCL))
            : fc_param_->output->mutable_data<float, cl::Buffer>(
                  TARGET(kOpenCL));
    // auto* out_buf =
    //     GET_BUFFER_GPU(fc_param_->output);
    printf("fc target %d\n", fc_param_->output->target());

    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_buf);
    CL_CHECK_FATAL(status);
    if (double_buffer_) {
      auto* w_buf = GET_BUFFER_GPU(w_gpu_t_);
      status = kernel.setArg(1, *w_buf);
      CL_CHECK_FATAL(status);
    } else {
      auto* w_img = DATA_GPU(w_gpu_t_);
      status = kernel.setArg(1, *w_img);
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
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
    // #ifdef LITE_WITH_PROFILE
    event_.wait();
    auto queue_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    auto submit_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    auto run_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto run_stop_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double time_ms = (submit_start_nanos - queue_start_nanos) / 1000000.0;
    std::cout << "0GetQueuedToSubmitTime: " << time_ms << std::endl;

    time_ms = (run_start_nanos - submit_start_nanos) / 1000000.0;
    std::cout << "0GetSubmitToStartTime: " << time_ms << std::endl;

    time_ms = (run_stop_nanos - run_start_nanos) / 1000000.0;
    std::cout << "0GetStartToEndTime: " << time_ms << std::endl;
    // #endif
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
  bool is_adreno_{true};
  bool double_buffer_{false};
  bool prepare_trans_w_{true};
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
    fc, kOpenCL, kFloat, kNCHW, paddle::lite::kernels::opencl::FcCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fc, kOpenCL, kFloat, kNCHW, paddle::lite::kernels::opencl::FcCompute, pc)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
