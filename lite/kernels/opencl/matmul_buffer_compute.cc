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

class MatMulV2Compute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::MatMulParam;

  void transpose_cpu(const float* in_data,
                     float* out_data,
                     const int in_rows,
                     const int in_cols) {
    CHECK(in_data && out_data && in_rows > 0 && in_cols > 0);
    for (int r = 0; r < in_rows; ++r) {
      for (int c = 0; c < in_cols; ++c) {
        out_data[c * in_rows + r] = in_data[r * in_cols + c];
      }
    }
  }

  void PrepareForRun() override {
    matmul_v2_param_ = param_.get_mutable<param_t>();
    transpose_x_ = matmul_v2_param_->transpose_X;
    transpose_y_ = matmul_v2_param_->transpose_Y;
    alpha_ = matmul_v2_param_->alpha;

    Tensor y_trans_cpu_t;
    auto y_t = matmul_v2_param_->Y;
    if (y_t->persistable() && transpose_y_) {
      LOG(INFO) << "y_t->persistable()";
      y_trans_cpu_t.Resize(y_t->dims());
      transpose_cpu(y_t->data<float>(),
                    y_trans_cpu_t.mutable_data<float>(),
                    y_t->dims()[0],
                    y_t->dims()[1]);
      y_t = &y_trans_cpu_t;
    }

    // upload y to gpu
    y_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    auto y_gpu_data =
        y_gpu_t_->mutable_data(TARGET(kOpenCL), y_t->memory_size());
    TargetWrapperCL::MemcpySync(
        y_gpu_data, y_t->raw_data(), y_t->memory_size(), IoDirection::HtoD);
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
        lda_ = m_;
      } else {
        m_ = x_dims[0];
        k_ = x_dims[1];
        lda_ = k_;
      }

      if (transpose_y_) {
        n_ = y_dims[0];
        ldb_ = n_;
      } else {
        n_ = y_dims[1];
        ldb_ = n_;
      }

      const auto out_dims = matmul_v2_param_->Out->dims();
      CHECK_EQ(m_, out_dims[0]);
      CHECK_EQ(n_, out_dims[1]);
      ldc_ = out_dims[1];

      const int x_k = k_;
      const int y_k = matmul_v2_param_->transpose_Y ? y_dims[1] : y_dims[0];
      CHECK(x_k == y_k);

#ifdef LITE_WITH_LOG
      LOG(INFO) << "x_dims:" << x_dims;
      LOG(INFO) << "y_dims:" << y_dims;
      LOG(INFO) << "transpose_X:" << transpose_x_;
      LOG(INFO) << "transpose_Y:" << transpose_y_;
      LOG(INFO) << "m_:" << m_ << ", k_:" << k_ << ", n_=" << n_;
      LOG(INFO) << "lda_:" << lda_ << ", ldb_:" << ldb_ << ", ldc_:" << ldc_;
#endif

      if (m_ == 1) {
        kernel_func_name_ = "gemv_1x4";
      } else {
        kernel_func_name_ = "mat_mul_naive";
      }
      VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
      auto& context = ctx_->As<OpenCLContext>();
      context.cl_context()->AddKernel(kernel_func_name_,
                                      "buffer/mat_mul_kernel.cl",
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
    if (kernel_func_name_ == "mat_mul_naive") {
      global_work_size_ =
          cl::NDRange{static_cast<size_t>(m_), static_cast<size_t>(n_)};
    } else if (kernel_func_name_ == "gemv_1x4") {
      global_work_size_ = cl::NDRange{static_cast<size_t>((n_ + 3) / 4)};
    }
  }

  void Run() override {
    auto* x_buf = matmul_v2_param_->X->data<float, cl::Buffer>();
    auto* y_buf = y_gpu_t_->template data<float, cl::Buffer>();
    auto* out_buf =
        matmul_v2_param_->Out->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *y_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, static_cast<const int>(m_));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, static_cast<const int>(n_));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, static_cast<const int>(k_));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(6, lda_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(7, ldb_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(8, ldc_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(9, alpha_);
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
  int m_{0};
  int n_{0};
  int k_{0};
  int lda_{0};
  int ldb_{0};
  int ldc_{0};
  bool transpose_x_{false};
  bool transpose_y_{false};
  float alpha_{1.0f};
  param_t* matmul_v2_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{"-DCL_DTYPE_float "};
  std::string time_stamp_{GetTimeStamp()};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;

  std::unique_ptr<Tensor> y_gpu_t_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_t_{nullptr};
  std::unique_ptr<Tensor> y_trans_cpu_t_{nullptr};
  std::unique_ptr<Tensor> y_trans_gpu_t_{nullptr};

  cl::NDRange global_work_size_;
  cl::Kernel kernel_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(matmul,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::MatMulV2Compute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul_v2,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::MatMulV2Compute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
