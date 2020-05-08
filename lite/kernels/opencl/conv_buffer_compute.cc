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

#include "lite/kernels/opencl/conv_buffer_compute.h"

#include <sstream>

#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void ConvCompute::PrepareForRun() {
  const auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);

  int bs = x_dims[0];
  int c_in = x_dims[1];
  int h_out = output_dims[2];
  int w_out = output_dims[3];
  int kernel_h = filter_dims[2];  // oihw
  int kernel_w = filter_dims[3];
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int groups = param.groups;
  bool relu_fused = param.fuse_relu;
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);
  bool zero_pad = (pad_h == 0) && (pad_w == 0);

  bool pad_equal =
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));

  VLOG(3) << "Is relu fused? / " << (relu_fused ? "Yes" : "No");
  VLOG(3) << "groups:" << groups << " stride_h:" << stride_h
          << " stride_w:" << stride_w << " pad_h:" << pad_h
          << " pad_w:" << pad_w << " kernel_h:" << kernel_h
          << " kernel_h:" << kernel_h;
  VLOG(3) << "x_dims:" << x_dims[0] << " " << x_dims[1] << " " << x_dims[2]
          << " " << x_dims[3];
  VLOG(3) << "output_dims:" << output_dims[0] << " " << output_dims[1] << " "
          << output_dims[2] << " " << output_dims[3];
  VLOG(3) << "filter_dims:" << filter_dims[0] << " " << filter_dims[1] << " "
          << filter_dims[2] << " " << filter_dims[3];

  if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 &&
      zero_pad && no_dilation && pad_equal) {
    // conv2d_1x1
    /* TODO(ysh329): CL_OUT_OF_MEMORY when use gemm_batched OpenCL kernel,
                 use gemm_batched_naive instead.
    kernel_func_names_.push_back("gemm_batch");
  */
    kernel_func_names_.push_back("gemm_batch_naive");
    kernel_func_paths_.push_back("buffer/fc_kernel.cl");
    if (relu_fused) {
      build_options_.push_back("-DCL_DTYPE_float -DRELU");
    } else if (param.activation_param.active_type ==
               lite_api::ActivationType::kRelu6) {
      build_options_.push_back("-DCL_DTYPE_float -DRELU6");
    } else {
      build_options_.push_back("-DCL_DTYPE_float");
    }
    impl_ = &ConvCompute::Conv2d1x1;
  } else if (pad_equal) {
    kernel_func_names_.push_back("im2col");
    /* TODO(ysh329): CL_OUT_OF_MEMORY when use gemm_batched OpenCL kernel,
                 use gemm_batched_naive instead.
    kernel_func_names_.push_back("gemm_batch");
  */
    kernel_func_names_.push_back("gemm_batch_naive");
    kernel_func_paths_.push_back("buffer/im2col_kernel.cl");
    kernel_func_paths_.push_back("buffer/fc_kernel.cl");
    build_options_.push_back("-DCL_DTYPE_float");
    if (relu_fused) {
      build_options_.push_back("-DCL_DTYPE_float -DRELU");
    } else if (param.activation_param.active_type ==
               lite_api::ActivationType::kRelu6) {
      build_options_.push_back("-DCL_DTYPE_float -DRELU6");
    } else {
      build_options_.push_back("-DCL_DTYPE_float");
    }
    impl_ = &ConvCompute::GemmlikeConv2d;
    col_buffer_.reset(new lite::Tensor);
    col_buffer_->Resize({bs, c_in, kernel_h * kernel_w, h_out * w_out});
    col_buffer_->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  } else {
    LOG(FATAL) << "This pad not support ! " << paddings[0] << ", "
               << paddings[1] << ", " << paddings[2] << ", " << paddings[3];
  }

  for (size_t i = 0; i < kernel_func_names_.size(); i++) {
    context.cl_context()->AddKernel(kernel_func_names_[i],
                                    kernel_func_paths_[i],
                                    build_options_[i],
                                    time_stamp_);
  }
}

void ConvCompute::GemmlikeConv2d() {
  const auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();
  int bs = x_dims[0];
  int c_in = x_dims[1];
  int h_in = x_dims[2];
  int w_in = x_dims[3];
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int c_out = output_dims[1];
  int h_out = output_dims[2];
  int w_out = output_dims[3];
  int kernel_h = filter_dims[2];
  int kernel_w = filter_dims[3];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dilation_h = dilations[0];
  int dilation_w = dilations[1];

  auto* x_buf = param.x->data<float, cl::Buffer>();
  auto* filter_buf = param.filter->data<float, cl::Buffer>();
  auto* bias_buf = (param.bias == nullptr)
                       ? static_cast<cl::Buffer*>(nullptr)
                       : param.bias->data<float, cl::Buffer>();
  auto* output_buf =
      param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto* col_buf = col_buffer_->mutable_data<float, cl::Buffer>();

  auto& context = ctx_->As<OpenCLContext>();
  std::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0] << time_stamp_;
  auto img2col_kernel = context.cl_context()->GetKernel(kernel_key.str());

  int n_threads = c_in * h_out * w_out;
  int in_stride = c_in * h_in * w_in;
  int out_stride = c_in * kernel_h * kernel_w * h_out * w_out;
  int img_offset = 0;
  int col_offset = 0;
  int arg_idx = 0;
  cl_int status;
  for (int b = 0; b < bs; b++) {
    img_offset = b * in_stride;
    col_offset = b * out_stride;
    arg_idx = 0;
    status = img2col_kernel.setArg(arg_idx, *x_buf);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, img_offset);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, n_threads);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, h_in);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, w_in);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, kernel_h);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, kernel_w);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, pad_h);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, pad_w);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, stride_h);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, stride_w);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, dilation_h);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, dilation_w);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, h_out);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, w_out);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, *col_buf);
    CL_CHECK_FATAL(status);
    status = img2col_kernel.setArg(++arg_idx, col_offset);
    CL_CHECK_FATAL(status);

    auto global_work_size = cl::NDRange{static_cast<size_t>(out_stride)};

    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        img2col_kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        nullptr);
    CL_CHECK_FATAL(status);
  }

  int m = c_out;
  int k = c_in * kernel_h * kernel_w;
  int n = h_out * w_out;
  VLOG(4) << "m = " << m << " n = " << n << " k = " << k;
  kernel_key.str("");
  kernel_key << kernel_func_names_[1] << build_options_[1] << time_stamp_;
  auto gemm_kernel = context.cl_context()->GetKernel(kernel_key.str());
  GemmBatched(
      gemm_kernel, col_buf, filter_buf, bias_buf, output_buf, bs, m, n, k);
}

void ConvCompute::Conv2d1x1() {
  const auto& param = *param_.get_mutable<param_t>();
  const int batch_size = param.x->dims()[0];
  const int k = param.x->dims()[1];  // K: input_channel
  const int n = param.x->dims()[2] *
                param.x->dims()[3];       // N == X_HxW == input_h * input_w
  const int m = param.output->dims()[1];  // M: output_channel == filter number

  VLOG(4) << "m = " << m << " n = " << n << " k = " << k;

  if (param.groups != 1) {
    LOG(FATAL) << "conv2d_1x1 with group > 1 not supported and param.groups = "
               << param.groups;
  }

  auto* x_d = param.x->data<float, cl::Buffer>();
  auto* filter_d = param.filter->data<float, cl::Buffer>();
  auto* bias_d = (param.bias == nullptr)
                     ? static_cast<cl::Buffer*>(nullptr)
                     : param.bias->data<float, cl::Buffer>();
  auto* output_d =
      param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

  auto& context = ctx_->As<OpenCLContext>();
  std::stringstream kernel_key;
  kernel_key << kernel_func_names_.front() << build_options_.front()
             << time_stamp_;
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());

  GemmBatched(kernel, x_d, filter_d, bias_d, output_d, batch_size, m, n, k);
}
// a: filter_d ==> <m, k> <=> <oc, ic>
// b: x_d      ==> <k, n> <=> <ic, ih*iw>
// c: output_d ==> <m, n> <=> <oc, ih*iw>
void ConvCompute::GemmBatched(cl::Kernel& kernel,
                              const cl::Buffer* x_d,
                              const cl::Buffer* filter_d,
                              const cl::Buffer* bias_d,
                              cl::Buffer* output_d,
                              const int batch_size,
                              const int m,
                              const int n,
                              const int k) {
  /* TODO(ysh329): CL_OUT_OF_MEMORY when use gemm_batch OpenCL kernel,
                   use gemm_batch_naive instead.
    auto global_work_size = cl::NDRange{static_cast<size_t>((m + 7) / 8),
                                        static_cast<size_t>((n + 3) / 4),
                                        static_cast<size_t>(batch_size)};
  */
  auto global_work_size = cl::NDRange{static_cast<size_t>(m),
                                      static_cast<size_t>(n),
                                      static_cast<size_t>(batch_size)};
  auto local_work_size = cl::NDRange{16, 16};  // cl::NullRange;

  auto& context = ctx_->As<OpenCLContext>();
  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, *filter_d);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *x_d);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *bias_d);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *output_d);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, m);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, n);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, k);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, batch_size);
  CL_CHECK_FATAL(status);

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      local_work_size,
      nullptr,
      nullptr);
  CL_CHECK_FATAL(status);
}

void ConvCompute::Run() { (this->*impl_)(); }

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::ConvCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
