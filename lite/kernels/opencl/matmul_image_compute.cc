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
  //  |  0  1  2  3 16 17 18 19 |
  //  |  4  5  6  7 20 21 22 23 |
  //  |  8  9 10 11 24 25 25 27 |
  //  | 12 13 14 15 28 29 30 31 |
  void RearrangeByBlk4x4(const float* src, void* dst, size_t O, size_t I) {
    bool fp16_support =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
    LOG(INFO) << "fp16_support = " << fp16_support;
    float* dst_fp32 = static_cast<float*>(dst);
    half_t* dst_fp16 = static_cast<half_t*>(dst);

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

    const int thres_k = 1024;
    bool precision_forced_to_fp32 = false;
    const bool enable_fp16 =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
    if (enable_fp16) {
      k_ = transpose_y_ ? y_dims[1] : y_dims[0];
      if (k_ > thres_k) {
        CLRuntime::Global()->set_precision(lite_api::CL_PRECISION_FP32);
        build_options_ += " -DCL_DTYPE_half -DCL_DTYPE_FLOAT_FORCE ";
        precision_forced_to_fp32 = true;
      }
    }
    int k_y = y_dims[0];
    n_ = y_dims[1];
    Tensor y_trans_cpu_t;
    LOG(INFO) << "persistableY: " << y_t->persistable()
              << ", tranposeY: " << transpose_y_;
    if (transpose_y_ && y_dims.size() >= 2) {
      y_trans_cpu_t.Resize(y_t->dims());
      transpose_cpu(y_t->data<float>(),
                    y_trans_cpu_t.mutable_data<float>(),
                    y_t->dims()[0],
                    y_t->dims()[1]);
      y_t = &y_trans_cpu_t;
      k_y = y_dims[1];
      n_ = y_dims[0];
    }
    auto y_cpu_t = std::unique_ptr<Tensor>(new Tensor);
    auto y_ext_dims = DDim(std::vector<DDim::value_type>{1, 1});
    if (y_dims.size() == 2) {
      y_ext_dims[0] = ROUND_UP(y_dims[0], 4);
      y_ext_dims[1] = ROUND_UP(y_dims[1], 4);
    } else if (y_dims.size() == 1 && (!transpose_y_)) {
      y_ext_dims[0] = ROUND_UP(y_dims[0], 4);
      y_ext_dims[1] = ROUND_UP(1, 4);
      n_ = 1;
    }
    y_cpu_t->Resize(y_ext_dims);
    auto* y_buffer_data = MUTABLE_DATA_CPU(y_cpu_t.get());
    auto* y_cpu = y_t->data<float>();
    RearrangeByBlk4x4(y_cpu, y_buffer_data, k_y, n_);

    y_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    auto y_gpu_data =
        y_gpu_t_->mutable_data(TARGET(kOpenCL), y_cpu_t->memory_size());
    TargetWrapperCL::MemcpySync(y_gpu_data,
                                y_cpu_t->raw_data(),
                                y_cpu_t->memory_size(),
                                IoDirection::HtoD);
    y_buf_ = GET_BUFFER_GPU(y_gpu_t_);

    // reset to original fp16 precision
    if (precision_forced_to_fp32) {
      CLRuntime::Global()->set_precision(lite_api::CL_PRECISION_FP16);
    }
  }

  void ReInitWhenNeeded() override {
    const auto x_dims = matmul_v2_param_->X->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;
      // compute m,n,k
      const auto y_dims = matmul_v2_param_->Y->dims();
      const auto out_dims = matmul_v2_param_->Out->dims();
#ifdef LITE_WITH_LOG
      LOG(INFO) << "x_dims:" << x_dims;
      LOG(INFO) << "y_dims:" << y_dims;
      LOG(INFO) << "out_dims:" << out_dims;
      LOG(INFO) << "transpose_X:" << transpose_x_;
      LOG(INFO) << "transpose_Y:" << transpose_y_;
#endif
      if (x_dims.size() == 2 && y_dims.size() == 2) {
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
      } else if (x_dims.size() == 1 && y_dims.size() == 1 &&
                 x_dims[0] == y_dims[0]) {
        CHECK_EQ(transpose_x_, false) << "unsupported when x_transpose is true";
        CHECK_EQ(transpose_y_, false) << "unsupported when y_transpose is true";
        m_ = 1, n_ = 1;
        k_ = y_dims[0];
      } else if (x_dims.size() == 1 && y_dims.size() == 1 &&
                 x_dims[0] != y_dims[0]) {
        CHECK_EQ(transpose_x_, true) << "unsupported when x_transpose is false";
        CHECK_EQ(transpose_y_, true) << "unsupported when y_transpose is false";
        m_ = x_dims[0], n_ = y_dims[0];
        k_ = 1;
      } else if (x_dims.size() == 4 && y_dims.size() == 1 &&
                 x_dims[x_dims.size() - 1] == y_dims[0]) {
        m_ = x_dims[0], n_ = x_dims.count(0, x_dims.size() - 1) / x_dims[0];
        k_ = y_dims[0];
      } else if (x_dims.size() > 2 && y_dims.size() >= 2) {
        // TODO(zhenlin-work)
      }

      CHECK_EQ(m_, out_dims[0]);
      CHECK_EQ(n_, out_dims[1]);

      k_blks_ = UP_DIV(k_, 4);
      n_blks_ = UP_DIV(n_, 4);
#ifdef LITE_WITH_LOG
      LOG(INFO) << "x_dims:" << x_dims;
      LOG(INFO) << "y_dims:" << y_dims;
      LOG(INFO) << "out_dims:" << out_dims;
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
    LOG(INFO) << "global_work_size[3D]: " << global_work_size_[0] << " "
              << global_work_size_[1] << " " << global_work_size_[2];
  }
  void Run() override {
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
  double GetStartToEndTime(const cl::Event& event) {
    auto start_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto stop_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return (stop_nanos - start_nanos) / 1000000.0;
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
  bool uselocalmem_{false};
  bool usetranspose_y_{false};
  float alpha_{1.0f};
  param_t* matmul_v2_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  std::unique_ptr<Tensor> y_gpu_t_{nullptr};
  const cl::Buffer* y_buf_{nullptr};

  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_;
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
