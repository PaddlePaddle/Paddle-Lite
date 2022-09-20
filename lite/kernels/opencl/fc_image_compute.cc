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

#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class FcImageCompute : public KernelLite<TARGET(kOpenCL),
                                         PRECISION(kFP16),
                                         DATALAYOUT(kImageFolder)> {
 public:
  void PrepareForRun() override {
    auto& param = this->Param<operators::FcParam>();
    const auto x_dims = param.input->dims();
    const auto out_dims = param.output->dims();
    const auto w_t = param.w;
    const auto bias_t = param.bias;
    has_bias_ = (bias_t == nullptr) ? false : true;

    // Runtime precision can be forced to fp32 to avoid the loss of accuracy
    // when K is larger than thres_k.
    // But this will increase the running time of fc because running time under
    // fp32 is two time longer than that under fp16.
    // So here we set thres_k higher as speed is the hightest priority by
    // default.
    const int thres_k = 1024;
    bool precision_forced_to_fp32 = false;
    const bool enable_fp16 =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
    if (enable_fp16) {
      k_ = x_dims.Slice(param.in_num_col_dims, x_dims.size()).production();
      if (k_ > thres_k) {
        CLRuntime::Global()->set_precision(lite_api::CL_PRECISION_FP32);
        build_options_ += " -DCL_DTYPE_half -DCL_DTYPE_FLOAT_FORCE ";
        precision_forced_to_fp32 = true;
      }
    }

    // convert weights from cpu to gpu
    auto w_cpu_t = std::unique_ptr<Tensor>(new Tensor);
    w_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    if (x_dims.size() > 2 && x_dims.size() <= 4) {
      layout_input_image_ = std::unique_ptr<Tensor>(new Tensor);
    }
    if (out_dims.size() > 2) {
      layout_output_image_ = std::unique_ptr<Tensor>(new Tensor);
    }

    const auto w_dims = w_t->dims();

    auto w_ext_dims = w_dims;
    w_ext_dims[0] = ROUND_UP(w_dims[0], 4);
    w_ext_dims[1] = ROUND_UP(w_dims[1], 4);
    w_cpu_t->Resize(w_ext_dims);
    auto* w_buffer_data = MUTABLE_DATA_CPU(w_cpu_t.get());
    size_t buf_size = w_cpu_t->memory_size();

    auto* w_cpu = param.w->mutable_data<float>();
    OI2OIO4I4(w_cpu, w_buffer_data, w_dims[0], w_dims[1]);

    auto* w_gpu_data =
        w_gpu_t_->mutable_data(TARGET(kOpenCL), w_cpu_t->memory_size());
    TargetWrapperCL::MemcpySync(w_gpu_data,
                                w_cpu_t->raw_data(),
                                w_cpu_t->memory_size(),
                                IoDirection::HtoD);
    w_buf_ = GET_BUFFER_GPU(w_gpu_t_);

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

    // reset to original fp16 precision
    if (precision_forced_to_fp32) {
      CLRuntime::Global()->set_precision(lite_api::CL_PRECISION_FP16);
    }
  }

  void ReInitWhenNeeded() override {
    auto& param = this->Param<operators::FcParam>();
    const auto x_dims = param.input->dims();
    const auto out_dims = param.output->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute m,n,k
      const auto w_dims = param.w->dims();
      CHECK_GE(x_dims.size(), 2UL);
      CHECK_LE(x_dims.size(), 4UL);
      CHECK_GE(w_dims.size(), 2UL);
      CHECK_LE(param.output->dims().size(), 4UL);

      int in_num_col_dims = param.in_num_col_dims;
      std::string op_type = param.op_type;
      if (op_type == "matmul" || op_type == "matmul_v2") {
        in_num_col_dims = x_dims.size() - 1;
      }
      m_ = x_dims.Slice(0, in_num_col_dims).production();
      k_ = x_dims.Slice(in_num_col_dims, x_dims.size()).production();
      n_ = w_dims[1];
      CHECK_EQ(k_, static_cast<int>(w_dims[0]));
      k_blks_ = UP_DIV(k_, 4);
      n_blks_ = UP_DIV(n_, 4);

      kernel_func_name_ = "fc";
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
      context.cl_context()->AddKernel(
          kernel_func_name_, "image/fc_kernel.cl", build_options_, time_stamp_);
      STL::stringstream kernel_key;
      kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
      kernel_ = context.cl_context()->GetKernel(kernel_key.str());

      if (x_dims.size() > 2 && x_dims.size() <= 4) {
        context.cl_context()->AddKernel(
            "input_layout", "image/fc_kernel.cl", build_options_, time_stamp_);
        STL::stringstream kernel_layout_key;
        kernel_layout_key << "input_layout" << build_options_ << time_stamp_;
        kernel_input_layout_ =
            context.cl_context()->GetKernel(kernel_layout_key.str());
      }

      if (out_dims.size() > 2) {
        context.cl_context()->AddKernel(
            "output_layout", "image/fc_kernel.cl", build_options_, time_stamp_);
        STL::stringstream kernel_output_key;
        kernel_output_key << "output_layout" << build_options_ << time_stamp_;
        kernel_output_layout_ =
            context.cl_context()->GetKernel(kernel_output_key.str());
      }

      SetGlobalLocalWorkSize();
    }
  }

  void Run() override {
    auto& param = this->Param<operators::FcParam>();
    auto x_dims = param.input->dims();
    auto out_dims = param.output->dims();
    cl::Image2D* x_img_src = DATA_GPU(param.input);

    if (x_dims.size() > 2 && x_dims.size() <= 4) {
      cl::NDRange layout_gws;
      int in_num_col_dims = param.in_num_col_dims;
      auto new_dims = Broadcast2GpuShape(x_dims);

      int N = new_dims[0];
      int C = new_dims[1];
      int H = new_dims[2];
      int W = new_dims[3];

      if (x_dims.size() == 3) {
        in_num_col_dims++;
      }

      x_img_ =
          MUTABLE_DATA_GPU(layout_input_image_, UP_DIV(k_, 4), m_, nullptr);
      int out_stride = 1;

      if (in_num_col_dims == 1) {
        out_stride = C * H * W;
        layout_gws = cl::NDRange{static_cast<size_t>((C * H * W + 3) / 4),
                                 static_cast<size_t>(N)};
      } else if (in_num_col_dims == 2) {
        out_stride = H * W;
        layout_gws = cl::NDRange{static_cast<size_t>((H * W + 3) / 4),
                                 static_cast<size_t>(N * C)};
      } else if (in_num_col_dims == 3) {
        out_stride = W;
        layout_gws = cl::NDRange{static_cast<size_t>((W + 3) / 4),
                                 static_cast<size_t>(N * C * H)};
      } else {
        LOG(FATAL) << "unsupport in_num_col_dims!";
      }

      auto& kernel = kernel_input_layout_;
      cl_int status;
      int arg_idx = 0;
      status = kernel.setArg(arg_idx++, *x_img_src);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, *x_img_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, W);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, H);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, W);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, W * H);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, W * H * C);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, out_stride);
      CL_CHECK_FATAL(status);

      auto& context = ctx_->As<OpenCLContext>();
      CHECK(context.cl_context() != nullptr);

      status = EnqueueNDRangeKernel(context,
                                    kernel,
                                    cl::NullRange,
                                    layout_gws,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
    } else {
      x_img_ = x_img_src;
    }
    if (out_dims.size() == 2) {
      out_img_src_ = MUTABLE_DATA_GPU(param.output, UP_DIV(n_, 4), m_, nullptr);
    } else {
      out_img_src_ =
          MUTABLE_DATA_GPU(layout_output_image_, UP_DIV(n_, 4), m_, nullptr);
    }

    auto& kernel = kernel_;
    cl_int status;
    int arg_idx = 0;
    status = kernel.setArg(arg_idx++, *x_img_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_img_src_);
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

    if (out_dims.size() > 2) {
      int N, C, H, W;
      cl::NDRange output_gws;

      if (out_dims.size() == 3) {
        N = 1;
        C = out_dims[0];
        H = out_dims[1];
        W = out_dims[2];
      } else {
        N = out_dims[0];
        C = out_dims[1];
        H = out_dims[2];
        W = out_dims[3];
      }
      output_gws = cl::NDRange{static_cast<size_t>((C + 3) / 4),
                               static_cast<size_t>(W),
                               static_cast<size_t>(N * H)};
      out_img_ =
          MUTABLE_DATA_GPU(param.output, UP_DIV(C, 4) * W, N * H, nullptr);

      auto& kernel = kernel_output_layout_;
      cl_int status;
      int arg_idx = 0;
      status = kernel.setArg(arg_idx++, *out_img_src_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, *out_img_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, W);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, H);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, C);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, H);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, W);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, W);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, W);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, H * W);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, C * H * W);
      CL_CHECK_FATAL(status);

      auto& context = ctx_->As<OpenCLContext>();
      CHECK(context.cl_context() != nullptr);

      status = EnqueueNDRangeKernel(context,
                                    kernel,
                                    cl::NullRange,
                                    output_gws,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
    }
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
  // This comment and impl refer to TFLite.
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
  cl::Kernel kernel_input_layout_;
  cl::Kernel kernel_output_layout_;
  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_;

  std::unique_ptr<Tensor> w_gpu_t_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_t_{nullptr};
  std::unique_ptr<Tensor> alpha_gpu_t_{nullptr};
  std::unique_ptr<Tensor> layout_input_image_{nullptr};
  std::unique_ptr<Tensor> layout_output_image_{nullptr};

  cl::Image2D* x_img_{nullptr};
  cl::Image2D* out_img_src_{nullptr};
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
                     kImageFolder,
                     paddle::lite::kernels::opencl::FcImageCompute,
                     image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageFolder))})
    .Finalize();
