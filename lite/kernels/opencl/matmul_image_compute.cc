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
                                               DATALAYOUT(kImageDefault)> {
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
    if (y_t->persistable() && (transpose_y_ || usetranspose_y_)) {
      y_trans_cpu_t.Resize(y_t->dims());
      transpose_cpu(y_t->data<float>(),
                    y_trans_cpu_t.mutable_data<float>(),
                    y_t->dims()[0],
                    y_t->dims()[1]);
      y_t = &y_trans_cpu_t;
    }
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

#ifdef LITE_WITH_LOG
      LOG(INFO) << "x_dims:" << x_dims;
      LOG(INFO) << "y_dims:" << y_dims;
      LOG(INFO) << "transpose_X:" << transpose_x_;
      LOG(INFO) << "transpose_Y:" << transpose_y_;
      LOG(INFO) << "m_:" << m_ << ", k_:" << k_ << ", n_=" << n_;
#endif
    }
  }

  void Run() override {
    y_buf_ = GET_BUFFER_GPU(y_gpu_t_);
    auto* x_img_ = GET_DATA_GPU(matmul_v2_param_->X);
    auto* out_img_ =
        MUTABLE_DATA_GPU(matmul_v2_param_->Out, UP_DIV(n_, 4), m_, nullptr);
    auto x_dims = matmul_v2_param_->X->dims();
    auto out_dims = matmul_v2_param_->Out->dims();
    size_t x_img_w, x_img_h, out_img_w, out_img_h;
    x_img_->getImageInfo(CL_IMAGE_WIDTH, &x_img_w);
    x_img_->getImageInfo(CL_IMAGE_HEIGHT, &x_img_h);
    out_img_->getImageInfo(CL_IMAGE_WIDTH, &out_img_w);
    out_img_->getImageInfo(CL_IMAGE_HEIGHT, &out_img_h);
    LOG(INFO) << "real input shape: " << x_img_w << " " << x_img_h;
    LOG(INFO) << "real output shape: " << out_img_w << " " << out_img_h;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    cl_int status;
    int arg_idx = 0;
    if (x_dims.size() == 2 && x_img_w == x_dims[1] &&
        out_dims[1] == out_img_w * 4) {
      if (uselocalmem_ == false) {
        kernel_func_name_ = "matmul_dimc";
        global_work_size_ = cl::NDRange{
            static_cast<cl::size_type>(out_img_w),
            static_cast<cl::size_type>(out_img_h),
        };
      } else {
        if (usetranspose_y_ == false) {
          kernel_func_name_ = "matmul_dim2_LM";
        } else {
          kernel_func_name_ = "matmul_dim2_LM_transY";
        }
        local_work_size_ = cl::NDRange(8, 8);
        global_work_size_ = cl::NDRange{
            static_cast<cl::size_type>(
                ROUND_UP(out_img_h, local_work_size_[0])),
            static_cast<cl::size_type>(out_img_w),
        };
      }
    }
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/matmul_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
    auto kernel = kernel_;
    status = kernel.setArg(arg_idx++, *x_img_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_img_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *y_buf_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, m_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, k_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, n_);
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
  bool transpose_x_{false};
  bool transpose_y_{false};
  bool uselocalmem_{false};
  bool usetranspose_y_{false};
  float alpha_{1.0f};
  param_t* matmul_v2_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  const cl::Buffer* y_buf_{nullptr};
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
                     kImageDefault,
                     paddle::lite::kernels::opencl::MatMulV2ImageCompute,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul_v2,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::MatMulV2ImageCompute,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
