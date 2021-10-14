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

class MatMulV2ImageCompute : public KernelLite<TARGET(kOpenCL),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kImageFolder)> {
 public:
  using param_t = operators::MatMulParam;

  void OI2OIO4I4(const float* src, void* dst, size_t O, size_t I) {
    bool fp16_support =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;

    float* dst_fp32 = static_cast<float*>(dst);
    half_t* dst_fp16 = static_cast<half_t*>(dst);
    // float* src_fp32 = static_cast<float*>(src);

    size_t i_blocks = UP_DIV(I, 4);
    size_t o_blocks = UP_DIV(O, 4);
    size_t dst_index = 0;
    for (size_t block_y = 0; block_y < o_blocks; block_y++) {
      for (size_t block_x = 0; block_x < i_blocks; block_x++) {
        for (size_t y_in_block = 0; y_in_block < 4; y_in_block++) {
          const int y = block_y * 4 + y_in_block;
          for (size_t x_in_block = 0; x_in_block < 4; x_in_block++) {
            const int x = block_x * 4 + x_in_block;
            if (y < O && x < I) {
              fp16_support ? dst_fp16[dst_index++] = Float2Half(src[y * I + x])
                           : dst_fp32[dst_index++] = src[y * I + x];
            } else {
              fp16_support ? dst_fp16[dst_index++] = Float2Half(0.f)
                           : dst_fp32[dst_index++] = 0.f;
            }
          }
        }
      }
    }
  }

  void PrepareForRun() override {
    matmul_v2_param_ = param_.get_mutable<param_t>();
    transpose_x_ = matmul_v2_param_->transpose_X;
    transpose_y_ = matmul_v2_param_->transpose_Y;
    alpha_ = matmul_v2_param_->alpha;

    auto y_t = matmul_v2_param_->Y;
    auto y_dims = y_t->dims();
    auto y_ext_dims = y_dims;
    y_ext_dims[0] = ROUND_UP(y_dims[0], 4);
    y_ext_dims[1] = ROUND_UP(y_dims[1], 4);
    auto y_cpu_t = std::unique_ptr<Tensor>(new Tensor);
    y_cpu_t->Resize(y_ext_dims);
    auto* y_buffer_data = MUTABLE_DATA_CPU(y_cpu_t.get());
    auto* y_cpu = y_t->data<float>();
    OI2OIO4I4(y_cpu, y_buffer_data, y_dims[0], y_dims[1]);

    y_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    auto y_gpu_data =
        y_gpu_t_->mutable_data(TARGET(kOpenCL), y_cpu_t->memory_size());
    TargetWrapperCL::MemcpySync(y_gpu_data,
                                y_cpu_t->raw_data(),
                                y_cpu_t->memory_size(),
                                IoDirection::HtoD);
  }

  void ReInitWhenNeeded() override {
    const auto x_dims = matmul_v2_param_->X->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute m,n,k
      const auto y_dims = matmul_v2_param_->Y->dims();
      CHECK_EQ(x_dims.size(), 2UL) << "Unsupported x_dims with " << x_dims;
      CHECK_EQ(y_dims.size(), 2UL) << "Unsupported y_dims with " << y_dims;
      CHECK_EQ(matmul_v2_param_->Out->dims().size(), 2UL);

      if (transpose_x_) {
        m_ = x_dims[1];
        k_ = x_dims[0];
      } else {
        m_ = x_dims[0];
        k_ = x_dims[1];
      }

      if (transpose_y_) {
        n_ = y_dims[0];
      } else {
        n_ = y_dims[1];
      }

      const auto out_dims = matmul_v2_param_->Out->dims();
      CHECK_EQ(m_, out_dims[0]);
      CHECK_EQ(n_, out_dims[1]);

      const int x_k = k_;
      const int y_k = matmul_v2_param_->transpose_Y ? y_dims[1] : y_dims[0];
      CHECK(x_k == y_k);

      k_blks_ = UP_DIV(k_, 4);
      n_blks_ = UP_DIV(n_, 4);
#ifdef LITE_WITH_LOG
      LOG(INFO) << "x_dims:" << x_dims;
      LOG(INFO) << "y_dims:" << y_dims;
      LOG(INFO) << "transpose_X:" << transpose_x_;
      LOG(INFO) << "transpose_Y:" << transpose_y_;
      LOG(INFO) << "m_:" << m_ << ", k_:" << k_ << ", n_=" << n_;
#endif
    }
    kernel_func_name_ = "fc";
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/fc_kernel.cl", build_options_, time_stamp_);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());

    SetGlobalLocalWorkSize();
  }

  void SetGlobalLocalWorkSize() {
    local_work_size_ = cl::NDRange(32, 4, 1);
    global_work_size_ = cl::NDRange(
        ROUND_UP(UP_DIV(n_, 4), local_work_size_[0]), local_work_size_[1], m_);
  }

  void Run() override {
    auto* y_buf_ = GET_BUFFER_GPU(y_gpu_t_);
    auto* x_img_ = GET_DATA_GPU(matmul_v2_param_->X);
    auto* out_img_ =
        MUTABLE_DATA_GPU(matmul_v2_param_->Out, UP_DIV(n_, 4), m_, nullptr);

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    cl_int status;
    int arg_idx = 0;
    auto kernel = kernel_;
    status = kernel.setArg(arg_idx++, *x_img_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_img_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *y_buf_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, m_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, k_blks_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, n_blks_);
    CL_CHECK_FATAL(status);

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
    ch->global_work_size = ch->NDRangeToStr(global_work_size_);
    ch->local_work_size = ch->NDRangeToStr(local_work_size_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  int m_{0};
  int n_{0};
  int k_{0};
  int k_blks_, n_blks_;
  bool transpose_x_{false};
  bool transpose_y_{false};
  float alpha_{1.0f};
  param_t* matmul_v2_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  std::unique_ptr<Tensor> y_gpu_t_{nullptr};

  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_{cl::NullRange};
  cl::Kernel kernel_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(matmul,
                     kOpenCL,
                     kFP16,
                     kImageFolder,
                     paddle::lite::kernels::opencl::MatMulV2ImageCompute,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageFolder))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul_v2,
                     kOpenCL,
                     kFP16,
                     kImageFolder,
                     paddle::lite::kernels::opencl::MatMulV2ImageCompute,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageFolder))})
    .Finalize();
