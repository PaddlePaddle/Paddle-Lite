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
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class FcImageCompute : public KernelLite<TARGET(kOpenCL),
                                         PRECISION(kFP16),
                                         DATALAYOUT(kImageDefault)> {
 public:
  void PrepareForRun() override {
    VLOG(4) << "0";
    auto& param = this->Param<operators::FcParam>();
    const auto w_t = param.w;
    const auto bias_t = param.bias;
    has_bias_ = (bias_t == nullptr) ? false : true;
    VLOG(4) << "0.1";

    // convert weights from cpu to gpu
    auto w_cpu_t = std::unique_ptr<Tensor>(new Tensor);
    w_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    const auto w_dims = w_t->dims();

    auto w_ext_dims = w_dims;
    w_ext_dims[0] = ROUND_UP(w_dims[0], 4);
    w_ext_dims[1] = ROUND_UP(w_dims[1], 4);
    w_cpu_t->Resize(w_ext_dims);
    VLOG(4) << "0.2";
    auto* w_buffer_data = MUTABLE_DATA_CPU(w_cpu_t.get());
    size_t buf_size = w_cpu_t->memory_size();

    VLOG(4) << "1";

    auto* w_cpu = param.w->mutable_data<float>();
    OI2OIO4I4(w_cpu, w_buffer_data, w_dims[0], w_dims[1]);
    VLOG(4) << "2";

    auto* w_gpu_data =
        w_gpu_t_->mutable_data(TARGET(kOpenCL), w_cpu_t->memory_size());
    VLOG(4) << "3";
    TargetWrapperCL::MemcpySync(w_gpu_data,
                                w_cpu_t->raw_data(),
                                w_cpu_t->memory_size(),
                                IoDirection::HtoD);
    w_buf_ = GET_BUFFER_GPU(w_gpu_t_);
    VLOG(4) << "4";
    // convert bias from cpu to gpu
    if (has_bias_) {
      auto bias_cpu_tensor = std::unique_ptr<Tensor>(new Tensor);
      bias_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      CLImageConverterFolder bias_converter;
      const DDim& bias_image_dims =
          bias_converter.InitImageDimInfoWith(param.bias->dims());
      bias_cpu_tensor->Resize({1, bias_image_dims[0], bias_image_dims[1], 4});
      auto* bias_image_data = MUTABLE_DATA_CPU(bias_cpu_tensor);
      auto* bias_cpu = param.bias->mutable_data<float>();
      bias_converter.NCHWToImage(bias_cpu, bias_image_data, param.bias->dims());
      MUTABLE_DATA_GPU(
          bias_gpu_t_, bias_image_dims[0], bias_image_dims[1], bias_image_data);

      build_options_ += " -DBIASE_CH";
      bias_img_ = DATA_GPU(bias_gpu_t_);
    }

    if (param.activation_type == "relu") {
      build_options_ += " -DRELU";
    } else if (param.activation_type == "relu6") {
      build_options_ += " -DRELU6";
    } else if (param.activation_type == "prelu") {
      std::string prelu_mode = param.Prelu_mode;
      build_options_ += " -DPRELU";
      if (prelu_mode == "channel") {
        build_options_ += " -DPRELU_CH";
      } else if (prelu_mode == "element") {
        build_options_ += " -DPRELU_ELE";
      } else {
        build_options_ += " -DPRELU_ALL";
      }

      // convert alpha from cpu to gpu
      auto alpha_cpu_tensor = std::unique_ptr<Tensor>(new Tensor);
      alpha_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      CLImageConverterFolder alpha_converter;
      const DDim& alpha_image_dims =
          alpha_converter.InitImageDimInfoWith(param.Prelu_alpha->dims());
      alpha_cpu_tensor->Resize(
          {1, alpha_image_dims[0], alpha_image_dims[1], 4});
      auto* alpha_image_data = MUTABLE_DATA_CPU(alpha_cpu_tensor);
      auto* alpha_cpu = param.Prelu_alpha->mutable_data<float>();
      alpha_converter.NCHWToImage(
          alpha_cpu, alpha_image_data, param.Prelu_alpha->dims());
      MUTABLE_DATA_GPU(alpha_gpu_t_,
                       alpha_image_dims[0],
                       alpha_image_dims[1],
                       alpha_image_data);
      alpha_img_ = DATA_GPU(alpha_gpu_t_);
    }
  }

  void ReInitWhenNeeded() override {
    auto& param = this->Param<operators::FcParam>();
    const auto x_dims = param.input->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute m,n,k
      const auto w_dims = param.w->dims();
      CHECK_GE(x_dims.size(), 2UL);
      CHECK_GE(w_dims.size(), 2UL);
      CHECK_EQ(param.output->dims().size(), 2UL);

      m_ = x_dims.Slice(0, param.in_num_col_dims).production();
      k_ = x_dims.Slice(param.in_num_col_dims, x_dims.size()).production();
      n_ = w_dims[1];
      CHECK_EQ(k_, static_cast<int>(w_dims[0]));
      k_blks_ = UP_DIV(k_, 4);
      n_blks_ = UP_DIV(n_, 4);

      kernel_func_name_ = "conv2d_1x1_fc";
#ifdef LITE_WITH_LOG
      VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
      VLOG(4) << "x_dims:" << x_dims;
      VLOG(4) << "w_dims:" << w_dims;
      if (has_bias_) {
        VLOG(4) << "bias_dims:" << param.bias->dims();
      }
      VLOG(4) << "M:" << m_ << " N:" << n_ << " K:" << k_;
#endif

      auto& context = ctx_->As<OpenCLContext>();
      context.cl_context()->AddKernel(kernel_func_name_,
                                      "image/conv2d_1x1_opt_kernel.cl",
                                      build_options_,
                                      time_stamp_);
      STL::stringstream kernel_key;
      kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
      kernel_ = context.cl_context()->GetKernel(kernel_key.str());

      SetGlobalLocalWorkSize();
    }
  }

  void Run() override {
    auto& param = this->Param<operators::FcParam>();

    x_img_ = DATA_GPU(param.input);
    out_img_ = MUTABLE_DATA_GPU(param.output, UP_DIV(n_, 4), m_, nullptr);

    auto& kernel = kernel_;
    cl_int status;
    int arg_idx = 0;
    status = kernel.setArg(arg_idx++, *x_img_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_img_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *w_buf_);
    CL_CHECK_FATAL(status);
    if (has_bias_) {
      status = kernel.setArg(arg_idx++, *bias_img_);
      CL_CHECK_FATAL(status);
    }
    if (alpha_img_) {
      status = kernel.setArg(arg_idx++, *alpha_img_);
      CL_CHECK_FATAL(status);
    }
    status = kernel.setArg(arg_idx++, m_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, k_blks_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, n_blks_);
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
    ch->global_work_size = ch->NDRangeToStr(global_work_size_);
    ch->local_work_size = ch->NDRangeToStr(local_work_size_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void SetGlobalLocalWorkSize() {
    local_work_size_ = cl::NDRange(32, 4, 1);
    global_work_size_ = cl::NDRange(
        ROUND_UP(UP_DIV(n_, 4), local_work_size_[0]), local_work_size_[1], m_);
  }

  // Change the travelsal order of the weight matrix in the following way:
  // The matrix is segmented to blocks of 4x4. If (any) dimension of the matrix
  // size is not divisible by 4, then pad with zeros. Each block is stored
  // contigously. The 16 elements within a block are ordered as 4 elements of
  // the first row, 4 elems of the second, etc. Blocks then traversed as
  // rows first, columns last. As an example, an 8x8 matrix would be traversed
  // as below.
  //
  //  |  0  1  2  3 16 17 18 19 |
  //  |  4  5  6  7 20 21 22 23 |
  //  |  8  9 10 11 24 25 25 27 |
  //  | 12 13 14 15 28 29 30 31 |
  //  | 32 33 34 35 48 49 50 51 |
  //  | 36 37 38 39 52 53 54 55 |
  //  | 40 41 42 43 56 57 58 69 |
  //  | 44 45 46 47 60 61 62 63 |
  //
  // The benefit of doing this is that reading contigous 16 elements gives a 4x4
  // block of the matrix, where the first 4 elements is the first row of the
  // block, second 4 elements is the second row of the block, etc. Subsequent
  // blocks contain elements of the same 4 columns.
  void OI2OIO4I4(void* src, void* dst, size_t O, size_t I) {
    bool fp16_support =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;

    float* dst_fp32 = static_cast<float*>(dst);
    half_t* dst_fp16 = static_cast<half_t*>(dst);
    float* src_fp32 = static_cast<float*>(src);

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
              fp16_support
                  ? dst_fp16[dst_index++] = Float2Half(src_fp32[y * I + x])
                  : dst_fp32[dst_index++] = src_fp32[y * I + x];
            } else {
              fp16_support ? dst_fp16[dst_index++] = Float2Half(0.f)
                           : dst_fp32[dst_index++] = 0.f;
            }
          }
        }
      }
    }
  }

 private:
  int m_, n_, k_, k_blks_, n_blks_;
  std::string kernel_func_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  DDim last_x_dims_;
  bool first_epoch_for_reinit_{true};
  bool has_bias_{false};

  cl::Kernel kernel_;
  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_;

  std::unique_ptr<Tensor> w_gpu_t_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_t_{nullptr};
  std::unique_ptr<Tensor> alpha_gpu_t_{nullptr};

  cl::Image2D* x_img_{nullptr};
  cl::Image2D* out_img_{nullptr};
  const cl::Buffer* w_buf_{nullptr};
  cl::Image2D* bias_img_{nullptr};
  cl::Image2D* alpha_img_{nullptr};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fc,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::FcImageCompute,
                     image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
