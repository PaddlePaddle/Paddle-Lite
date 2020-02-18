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

#include "lite/kernels/opencl/conv_compute.h"

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
    kernel_func_names_.push_back("gemm_batch");
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
    kernel_func_names_.push_back("gemm_batch");
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
    context.cl_context()->AddKernel(
        kernel_func_names_[i], kernel_func_paths_[i], build_options_[i]);
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
  kernel_key << kernel_func_names_[0] << build_options_[0];
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
  kernel_key << kernel_func_names_[1] << build_options_[1];
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
  kernel_key << kernel_func_names_.front() << build_options_.front();
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
  auto global_work_size = cl::NDRange{static_cast<size_t>((m + 7) / 8),
                                      static_cast<size_t>((n + 3) / 4),
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
      event_.get());
  CL_CHECK_FATAL(status);

  context.cl_wait_list()->emplace(output_d, event_);
}

void ConvCompute::Run() { (this->*impl_)(); }

/* image kernel*/
void ConvImageCompute::PrepareForRun() {
  const auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();

  float* filter_cpu = param.filter->mutable_data<float>();
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
      ((paddings[0] == paddings[1]) && (paddings[1] == paddings[2]) &&
       (paddings[2] == paddings[3]));
  bool stride_equal = stride_h == stride_w;
  bool dilation_equal = dilations[0] == dilations[1];

  CHECK(pad_equal && stride_equal && dilation_equal);

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
  if (kernel_h == 1 && kernel_w == 1) {
    // conv2d_1x1
    if (param.x->dims()[1] % 4 == 0) {
      kernel_func_names_.push_back("conv2d_1x1_simple");
    } else {
      kernel_func_names_.push_back("conv2d_1x1");
    }
    kernel_func_paths_.push_back("image/conv2d_1x1_kernel.cl");

    CLImageConverterNWBlock converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    std::vector<float> filter_image_v(filter_image_dims[0] *
                                      filter_image_dims[1] * 4);  // 4 : RGBA
    converter.NCHWToImage(filter_cpu, filter_image_v.data(), filter_dims);
    filter_gpu_image_.mutable_data<float, cl::Image2D>(
        filter_image_dims[0], filter_image_dims[1], filter_image_v.data());

    impl_ = &ConvImageCompute::Conv2d1x1;
#if 1  // TODO(ysh329): enable general dwconv
  } else if (filter_dims[1] == 1 && x_dims[1] == output_dims[1]) {
#else  // TODO(ysh329): remove dwconv3x3s1 and dwconv3x3 temporarily, need fix
  } else if (filter_dims[1] == 1 && x_dims[1] == output_dims[1] &&
             kernel_h == 3 && kernel_w == 3 && groups > 1) {
    // depth_conv2d_3x3s1, depth_conv2d_3x3
    if (stride_h == 1 && dilations[0] == 1) {
      kernel_func_names_.push_back("depth_conv2d_3x3s1");
      impl_ = &ConvImageCompute::DepthwiseConv2d3x3s1;
    } else {
      kernel_func_names_.push_back("depth_conv2d_3x3");
      impl_ = &ConvImageCompute::DepthwiseConv2d3x3;
    }
    kernel_func_paths_.push_back("image/depthwise_conv2d_kernel.cl");

    CLImageConverterNWBlock converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    std::vector<float> filter_image_v(filter_image_dims[0] *
                                      filter_image_dims[1] * 4);  // 4 : RGBA
    converter.NCHWToImage(filter_cpu, filter_image_v.data(), filter_dims);
    filter_gpu_image_.mutable_data<float, cl::Image2D>(
        filter_image_dims[0], filter_image_dims[1], filter_image_v.data());
  } else if (filter_dims[1] == 1 && x_dims[1] == output_dims[1] &&
             kernel_h != 3) {
#endif
    // depth_conv2d
    kernel_func_names_.push_back("depth_conv2d");
    kernel_func_paths_.push_back("image/depthwise_conv2d_basic_kernel.cl");

    CLImageConverterNWBlock converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    std::vector<float> filter_image_v(filter_image_dims[0] *
                                      filter_image_dims[1] * 4);  // 4 : RGBA
    converter.NCHWToImage(filter_cpu, filter_image_v.data(), filter_dims);
    filter_gpu_image_.mutable_data<float, cl::Image2D>(
        filter_image_dims[0], filter_image_dims[1], filter_image_v.data());

    impl_ = &ConvImageCompute::DepthwiseConv2d;
  } else if (kernel_h == 3 && kernel_h == 3) {
    // conv2d_3x3
    kernel_func_names_.push_back("conv2d_3x3");
    kernel_func_paths_.push_back("image/conv2d_3x3_kernel.cl");

    CLImageConverterFolder converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    std::vector<float> filter_image_v(filter_image_dims[0] *
                                      filter_image_dims[1] * 4);  // 4 : RGBA
    converter.NCHWToImage(filter_cpu, filter_image_v.data(), filter_dims);
    filter_gpu_image_.mutable_data<float, cl::Image2D>(
        filter_image_dims[0], filter_image_dims[1], filter_image_v.data());

    impl_ = &ConvImageCompute::Conv2d3x3;
  } else if (kernel_h == 5 && kernel_w == 5) {
    // conv2d_5x5
    kernel_func_names_.push_back("conv2d_5x5");
    kernel_func_paths_.push_back("image/conv2d_5x5_kernel.cl");

    CLImageConverterFolder converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    std::vector<float> filter_image_v(filter_image_dims[0] *
                                      filter_image_dims[1] * 4);  // 4 : RGBA
    converter.NCHWToImage(filter_cpu, filter_image_v.data(), filter_dims);
    filter_gpu_image_.mutable_data<float, cl::Image2D>(
        filter_image_dims[0], filter_image_dims[1], filter_image_v.data());

    impl_ = &ConvImageCompute::Conv2d5x5;
  } else if (kernel_h == 7 && kernel_w == 7) {
    // conv2d_7x7
    kernel_func_names_.push_back("conv2d_7x7");
    kernel_func_paths_.push_back("image/conv2d_7x7_kernel.cl");

    CLImageConverterFolder converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    std::vector<float> filter_image_v(filter_image_dims[0] *
                                      filter_image_dims[1] * 4);  // 4 : RGBA
    converter.NCHWToImage(filter_cpu, filter_image_v.data(), filter_dims);
    this->filter_gpu_image_.mutable_data<float, cl::Image2D>(
        filter_image_dims[0], filter_image_dims[1], filter_image_v.data());

    impl_ = &ConvImageCompute::Conv2d7x7;
  } else {
    LOG(FATAL) << "conv image compute not support this condition yet! ";
  }
  VLOG(1) << "kernel_func_names_[0]:" << kernel_func_names_[0]
          << " kernel_func_paths_[0]:" << kernel_func_paths_[0];

  std::string build_options_single(" -DCL_DTYPE_float");
  // relu options
  if (relu_fused) {
    build_options_single += " -DRELU";
  } else if (param.activation_param.active_type ==
             lite_api::ActivationType::kRelu6) {
    build_options_single += " -DRELU6";
  } else {
    // do nothing
  }
  // bias options
  const bool has_bias = param.bias != nullptr;
  const bool is_element_wise_bias =
      has_bias && param.output->dims() == param.bias->dims();
  if (has_bias) {
    build_options_single +=
        is_element_wise_bias ? " -DBIASE_ELE" : " -DBIASE_CH";

    // convert cpu buffer bias --> gpu image
    CLImageConverterFolder bias_converter;
    const DDim& bias_image_dims =
        bias_converter.InitImageDimInfoWith(param.bias->dims());
    std::vector<float> bias_image_v(bias_image_dims[0] * bias_image_dims[1] *
                                    4);
    float* bias_cpu_data = param.bias->mutable_data<float>();
    bias_converter.NCHWToImage(
        bias_cpu_data, bias_image_v.data(), param.bias->dims());
    this->bias_gpu_image_.mutable_data<float, cl::Image2D>(
        bias_image_dims[0], bias_image_dims[1], bias_image_v.data());
    // convert cpu buffer bias --> gpu image --- end ----
  }

  build_options_.push_back(build_options_single);

  for (size_t i = 0; i < kernel_func_names_.size(); i++) {
    context.cl_context()->AddKernel(
        kernel_func_names_[i], kernel_func_paths_[i], build_options_[i]);
  }
}

void ConvImageCompute::Conv2d1x1() {
  const auto& param = *param_.get_mutable<param_t>();
  auto input_dims = param.x->dims();
  auto paddings = *param.paddings;
  auto strides = param.strides;
  auto* input_image = param.x->data<float, cl::Image2D>();
  auto* filter_image = filter_gpu_image_.data<float, cl::Image2D>();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();

  int input_width = input_dims[3];
  int input_height = input_dims[2];
  int output_width = output_dims[3];
  int output_height = output_dims[2];
  auto out_image_shape = InitImageDimInfoWith(output_dims);
  auto* out_image = param.output->mutable_data<float, cl::Image2D>(
      out_image_shape["width"], out_image_shape["height"]);

  const bool has_bias = param.bias != nullptr;
  const bool is_element_wise_bias =
      has_bias && param.output->dims() == param.bias->dims();
  int offset = static_cast<int>(param.filter->dims()[2]) / 2 -
               static_cast<int>(paddings[0]);

  // calc input_c_block
  auto input_image_shape = InitImageDimInfoWith(input_dims);
  int input_c_block = input_image_shape["width"] / input_dims[3];
  int input_c = input_dims[1];
  auto dilations = *param.dilations;

  const std::vector<size_t>& default_work_size =
      DefaultWorkSize(output_dims,
                      DDim(std::vector<DDim::value_type>{
                          static_cast<int64_t>(out_image_shape["width"]),
                          static_cast<int64_t>(out_image_shape["height"])}));

  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];

  VLOG(4) << "============ conv2d_1x1 params ============";
  VLOG(4) << "input_image_shape: " << input_image_shape["width"] << ","
          << input_image_shape["height"];
  VLOG(4) << "input_c_block: " << input_c_block;
  VLOG(4) << "input_c: " << input_c;
  VLOG(4) << "input_image: " << input_image;
  VLOG(4) << "filter_dims: " << filter_dims;
  VLOG(4) << "filter_image: " << filter_image;
  VLOG(4) << "output_dims: " << output_dims;
  VLOG(4) << "out_image_shape: " << out_image_shape["width"] << ", "
          << out_image_shape["height"];
  VLOG(4) << "paddings: " << paddings[0] << "," << paddings[1];
  VLOG(4) << "has bias: " << has_bias;
  VLOG(4) << "is_element_wise_bias : " << is_element_wise_bias;
  VLOG(4) << "strides: " << strides[0] << "," << strides[1];
  VLOG(4) << "offset: " << offset;
  VLOG(4) << "dilations.size : " << dilations.size();
  VLOG(4) << "dilations: " << dilations[0] << ", " << dilations[1];
  VLOG(4) << "default work size{c_block, w, nh}: "
          << "{" << c_block << ", " << w << ", " << nh << ""
          << "}";

  CHECK_GE(dilations.size(), 2);
  CHECK(dilations[0] == dilations[1]);
  CHECK_GE(input_dims.size(), 4);
  CHECK_GE(paddings.size(), 2);
  CHECK(paddings[0] == paddings[1]);
  CHECK_GE(strides.size(), 2);
  CHECK(strides[0] == strides[1]);

  // handle bias  use buffer for channel wise , use image for element wise
  const cl::Buffer* bias_buf = nullptr;
  const cl::Image2D* bias_image = nullptr;
  if (has_bias) {
    bias_image = bias_gpu_image_.data<float, cl::Image2D>();
  }

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  std::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0];
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());
  int maped_w = maptofactor(w, 4);

  VLOG(4) << "kernel_key: " << kernel_key.str();
  VLOG(4) << "kernel ready ... " << kernel_key.str();
  VLOG(4) << "maped_w: " << maped_w;
  VLOG(4) << "hasbias: " << has_bias;

  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, c_block);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, maped_w);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, nh);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *input_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *filter_image);
  CL_CHECK_FATAL(status);
  if (has_bias) {
    status = kernel.setArg(++arg_idx, *bias_image);
    CL_CHECK_FATAL(status);
  }
  status = kernel.setArg(++arg_idx, *out_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, strides[0]);
  CL_CHECK_FATAL(status);

  status = kernel.setArg(++arg_idx, offset);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_c_block);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_c);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, dilations[0]);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_height);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_height);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, w);
  CL_CHECK_FATAL(status);

  auto global_work_size =
      cl::NDRange{static_cast<size_t>(default_work_size.data()[0]),
                  static_cast<size_t>(maped_w),
                  static_cast<size_t>(default_work_size.data()[2])};

  VLOG(4) << "out_image: " << out_image;
  VLOG(4) << "global_work_size[3D]: {" << global_work_size[0] << ","
          << global_work_size[1] << "," << global_work_size[2] << "}";

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      cl::NullRange,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);
  context.cl_wait_list()->emplace(out_image, event_);
}

void ConvImageCompute::Conv2d3x3() {
  const auto& param = *param_.get_mutable<param_t>();
  auto input_dims = param.x->dims();
  auto paddings = *param.paddings;
  auto strides = param.strides;

  auto* input_image = param.x->data<float, cl::Image2D>();
  auto* filter_image = filter_gpu_image_.data<float, cl::Image2D>();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();

  int input_width = input_dims[3];
  int input_height = input_dims[2];
  int input_channel = input_dims[1];
  int output_width = output_dims[3];
  int output_height = output_dims[2];
  int output_channel = output_dims[1];
  int filter_width = filter_dims[3];
  int filter_height = filter_dims[2];
  int filter_channel = filter_dims[1];
  auto out_image_shape = InitImageDimInfoWith(output_dims);
  auto* out_image = param.output->mutable_data<float, cl::Image2D>(
      out_image_shape["width"], out_image_shape["height"]);

  const bool has_bias = param.bias != nullptr;
  const bool is_element_wise_bias =
      has_bias && param.output->dims() == param.bias->dims();
  int offset = static_cast<int>(param.filter->dims()[2]) / 2 -
               static_cast<int>(paddings[0]);

  // calc input_c_block
  auto input_image_shape = InitImageDimInfoWith(input_dims);
  int input_c_block = input_image_shape["width"] / input_dims[3];
  int input_c = input_dims[1];
  auto dilations = *param.dilations;

  // re-calc group
  int new_groups{param.groups};
  if (filter_dims[0] == output_dims[1] && filter_dims[1] == input_dims[1]) {
    new_groups = 1;
  } else if (!(filter_dims[0] == input_dims[1] && filter_dims[1] == 1)) {
    new_groups = input_channel / filter_channel;
  }
  /* TODO(ysh329): mobile has no case below
     else {
      LOG(FATAL) << "Not support conv3x3 case with"
                 << " input_dims:" << input_dims << " output_dims:" <<
    output_dims
                 << " filter_dims:" << filter_dims;
    }
  */

  const std::vector<size_t>& default_work_size =
      DefaultWorkSize(output_dims,
                      DDim(std::vector<DDim::value_type>{
                          static_cast<int64_t>(out_image_shape["width"]),
                          static_cast<int64_t>(out_image_shape["height"])}));

  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];

  VLOG(4) << "============ conv2d params ============";
  VLOG(4) << "input_image_shape: " << input_image_shape["width"] << ","
          << input_image_shape["height"];
  VLOG(4) << "input_c_block: " << input_c_block;
  VLOG(4) << "input_c: " << input_c;
  VLOG(4) << "input_image: " << input_image;
  VLOG(4) << "input_dims: " << input_dims;
  VLOG(4) << "filter_dims: " << filter_dims;
  VLOG(4) << "filter_image: " << filter_image;
  VLOG(4) << "output_dims: " << output_dims;
  VLOG(4) << "out_image_shape: " << out_image_shape["width"] << ", "
          << out_image_shape["height"];
  VLOG(4) << "paddings: " << paddings[0] << "," << paddings[1];
  VLOG(4) << "has bias: " << has_bias;
  VLOG(4) << "is_element_wise_bias : " << is_element_wise_bias;
  VLOG(4) << "strides: " << strides[0] << "," << strides[1];
  VLOG(4) << "offset: " << offset;
  VLOG(4) << "dilations.size : " << dilations.size();
  VLOG(4) << "dilations: " << dilations[0] << ", " << dilations[1];
  VLOG(4) << "param.groups(groups):" << param.groups;
  VLOG(4) << "new_groups:" << new_groups;
  VLOG(4) << "default work size{c_block, w, nh}: "
          << "{" << c_block << ", " << w << ", " << nh << ""
          << "}";

  CHECK_GE(dilations.size(), 2);
  CHECK(dilations[0] == dilations[1]);
  CHECK_GE(input_dims.size(), 4);
  CHECK_GE(paddings.size(), 2);
  CHECK(paddings[0] == paddings[1]);
  CHECK_GE(strides.size(), 2);
  CHECK(strides[0] == strides[1]);

  const cl::Image2D* bias_image = nullptr;
  if (has_bias) {
    bias_image = bias_gpu_image_.data<float, cl::Image2D>();
  }

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  STL::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0];
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());
  VLOG(4) << "kernel_key: " << kernel_key.str();
  VLOG(4) << "kernel ready ... " << kernel_key.str();
  VLOG(4) << "w: " << w;

  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, c_block);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, w);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, nh);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *input_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *filter_image);
  CL_CHECK_FATAL(status);
  if (has_bias) {
    VLOG(4) << "set bias_image: ";
    status = kernel.setArg(++arg_idx, *bias_image);
    CL_CHECK_FATAL(status);
  }
  status = kernel.setArg(++arg_idx, *out_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, strides[0]);
  CL_CHECK_FATAL(status);

  status = kernel.setArg(++arg_idx, offset);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_c_block);
  CL_CHECK_FATAL(status);

  status = kernel.setArg(++arg_idx, dilations[0]);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_height);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_height);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_channel);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, filter_channel);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, filter_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, filter_height);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, new_groups);
  CL_CHECK_FATAL(status);

  auto global_work_size =
      cl::NDRange{static_cast<size_t>(default_work_size.data()[0]),
                  static_cast<size_t>(default_work_size.data()[1]),
                  static_cast<size_t>(default_work_size.data()[2])};

  VLOG(4) << "out_image: " << out_image;
  VLOG(4) << "global_work_size[3D]: {" << global_work_size[0] << ","
          << global_work_size[1] << "," << global_work_size[2] << "}";

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      cl::NullRange,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);
  context.cl_wait_list()->emplace(out_image, event_);
}

void ConvImageCompute::Conv2d5x5() {
  const auto& param = *param_.get_mutable<param_t>();
  auto input_dims = param.x->dims();
  auto paddings = *param.paddings;
  auto strides = param.strides;
  auto* input_image = param.x->data<float, cl::Image2D>();
  auto* filter_image = filter_gpu_image_.data<float, cl::Image2D>();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();

  int input_width = input_dims[3];
  int input_height = input_dims[2];
  int output_width = output_dims[3];
  int output_height = output_dims[2];
  int filter_width = filter_dims[3];
  int filter_height = filter_dims[2];
  auto out_image_shape = InitImageDimInfoWith(output_dims);
  auto* out_image = param.output->mutable_data<float, cl::Image2D>(
      out_image_shape["width"], out_image_shape["height"]);

  const bool has_bias = param.bias != nullptr;
  const bool is_element_wise_bias =
      has_bias && param.output->dims() == param.bias->dims();
  int offset = static_cast<int>(param.filter->dims()[2]) / 2 -
               static_cast<int>(paddings[0]);

  // calc input_c_block
  auto input_image_shape = InitImageDimInfoWith(input_dims);
  int input_c_block = input_image_shape["width"] / input_dims[3];
  int input_c = input_dims[1];
  auto dilations = *param.dilations;

  const std::vector<size_t>& default_work_size =
      DefaultWorkSize(output_dims,
                      DDim(std::vector<DDim::value_type>{
                          static_cast<int64_t>(out_image_shape["width"]),
                          static_cast<int64_t>(out_image_shape["height"])}));

  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];

  VLOG(4) << "============ conv2d params ============";
  VLOG(4) << "input_image_shape: " << input_image_shape["width"] << ","
          << input_image_shape["height"];
  VLOG(4) << "input_c_block: " << input_c_block;
  VLOG(4) << "input_c: " << input_c;
  VLOG(4) << "input_image: " << input_image;
  VLOG(4) << "input_dims: " << input_dims;
  VLOG(4) << "filter_dims: " << filter_dims;
  VLOG(4) << "filter_image: " << filter_image;
  VLOG(4) << "output_dims: " << output_dims;
  VLOG(4) << "out_image_shape: " << out_image_shape["width"] << ", "
          << out_image_shape["height"];
  VLOG(4) << "paddings: " << paddings[0] << "," << paddings[1];
  VLOG(4) << "has bias: " << has_bias;
  VLOG(4) << "is_element_wise_bias : " << is_element_wise_bias;
  VLOG(4) << "strides: " << strides[0] << "," << strides[1];
  VLOG(4) << "offset: " << offset;
  VLOG(4) << "dilations.size : " << dilations.size();
  VLOG(4) << "dilations: " << dilations[0] << ", " << dilations[1];
  VLOG(4) << "default work size{c_block, w, nh}: "
          << "{" << c_block << ", " << w << ", " << nh << ""
          << "}";

  CHECK_GE(dilations.size(), 2);
  CHECK(dilations[0] == dilations[1]);
  CHECK_GE(input_dims.size(), 4);
  CHECK_GE(paddings.size(), 2);
  CHECK(paddings[0] == paddings[1]);
  CHECK_GE(strides.size(), 2);
  CHECK(strides[0] == strides[1]);

  const cl::Image2D* bias_image = nullptr;
  if (has_bias) {
    bias_image = bias_gpu_image_.data<float, cl::Image2D>();
  }

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  STL::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0];
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());
  VLOG(4) << "kernel_key: " << kernel_key.str();
  VLOG(4) << "kernel ready ... " << kernel_key.str();
  VLOG(4) << "w: " << w;

  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, c_block);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, w);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, nh);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *input_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *filter_image);
  CL_CHECK_FATAL(status);
  if (has_bias) {
    VLOG(4) << "set bias_image: ";
    status = kernel.setArg(++arg_idx, *bias_image);
    CL_CHECK_FATAL(status);
  }
  status = kernel.setArg(++arg_idx, *out_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, strides[0]);
  CL_CHECK_FATAL(status);

  status = kernel.setArg(++arg_idx, offset);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_c_block);
  CL_CHECK_FATAL(status);

  status = kernel.setArg(++arg_idx, dilations[0]);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_height);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_height);
  CL_CHECK_FATAL(status);

  auto global_work_size =
      cl::NDRange{static_cast<size_t>(default_work_size.data()[0]),
                  static_cast<size_t>(default_work_size.data()[1]),
                  static_cast<size_t>(default_work_size.data()[2])};

  VLOG(4) << "out_image: " << out_image;
  VLOG(4) << "global_work_size[3D]: {" << global_work_size[0] << ","
          << global_work_size[1] << "," << global_work_size[2] << "}";

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      cl::NullRange,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);
  context.cl_wait_list()->emplace(out_image, event_);
}

void ConvImageCompute::Conv2d7x7() {
  const auto& param = *param_.get_mutable<param_t>();
  auto input_dims = param.x->dims();
  auto paddings = *param.paddings;
  auto strides = param.strides;
  auto* input_image = param.x->data<float, cl::Image2D>();
  auto* filter_image = filter_gpu_image_.data<float, cl::Image2D>();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();

  int input_width = input_dims[3];
  int input_height = input_dims[2];
  int output_width = output_dims[3];
  int output_height = output_dims[2];
  int filter_width = filter_dims[3];
  int filter_height = filter_dims[2];
  auto out_image_shape = InitImageDimInfoWith(output_dims);
  auto* out_image = param.output->mutable_data<float, cl::Image2D>(
      out_image_shape["width"], out_image_shape["height"]);

  const bool has_bias = param.bias != nullptr;
  const bool is_element_wise_bias =
      has_bias && param.output->dims() == param.bias->dims();
  int offset = static_cast<int>(param.filter->dims()[2]) / 2 -
               static_cast<int>(paddings[0]);

  // calc input_c_block
  auto input_image_shape = InitImageDimInfoWith(input_dims);
  int input_c_block = input_image_shape["width"] / input_dims[3];
  int input_c = input_dims[1];
  auto dilations = *param.dilations;

  const std::vector<size_t>& default_work_size =
      DefaultWorkSize(output_dims,
                      DDim(std::vector<DDim::value_type>{
                          static_cast<int64_t>(out_image_shape["width"]),
                          static_cast<int64_t>(out_image_shape["height"])}));

  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];

  VLOG(4) << "============ conv2d params ============";
  VLOG(4) << "input_image_shape: " << input_image_shape["width"] << ","
          << input_image_shape["height"];
  VLOG(4) << "input_c_block: " << input_c_block;
  VLOG(4) << "input_c: " << input_c;
  VLOG(4) << "input_image: " << input_image;
  VLOG(4) << "input_dims: " << input_dims;
  VLOG(4) << "filter_dims: " << filter_dims;
  VLOG(4) << "filter_image: " << filter_image;
  VLOG(4) << "output_dims: " << output_dims;
  VLOG(4) << "out_image_shape: " << out_image_shape["width"] << ", "
          << out_image_shape["height"];
  VLOG(4) << "paddings: " << paddings[0] << "," << paddings[1];
  VLOG(4) << "has bias: " << has_bias;
  VLOG(4) << "is_element_wise_bias : " << is_element_wise_bias;
  VLOG(4) << "strides: " << strides[0] << "," << strides[1];
  VLOG(4) << "offset: " << offset;
  VLOG(4) << "dilations.size : " << dilations.size();
  VLOG(4) << "dilations: " << dilations[0] << ", " << dilations[1];
  VLOG(4) << "default work size{c_block, w, nh}: "
          << "{" << c_block << ", " << w << ", " << nh << ""
          << "}";

  CHECK_GE(dilations.size(), 2);
  CHECK(dilations[0] == dilations[1]);
  CHECK_GE(input_dims.size(), 4);
  CHECK_GE(paddings.size(), 2);
  CHECK(paddings[0] == paddings[1]);
  CHECK_GE(strides.size(), 2);
  CHECK(strides[0] == strides[1]);

  const cl::Image2D* bias_image = nullptr;
  if (has_bias) {
    bias_image = bias_gpu_image_.data<float, cl::Image2D>();
  }

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  STL::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0];
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());
  VLOG(4) << "kernel_key: " << kernel_key.str();
  VLOG(4) << "kernel ready ... " << kernel_key.str();
  VLOG(4) << "w: " << w;

  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, c_block);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, w);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, nh);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *input_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *filter_image);
  CL_CHECK_FATAL(status);
  if (has_bias) {
    VLOG(4) << "set bias_image: ";
    status = kernel.setArg(++arg_idx, *bias_image);
    CL_CHECK_FATAL(status);
  }
  status = kernel.setArg(++arg_idx, *out_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, strides[0]);
  CL_CHECK_FATAL(status);

  status = kernel.setArg(++arg_idx, offset);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_c_block);
  CL_CHECK_FATAL(status);

  status = kernel.setArg(++arg_idx, dilations[0]);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_height);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_height);
  CL_CHECK_FATAL(status);

  auto global_work_size =
      cl::NDRange{static_cast<size_t>(default_work_size.data()[0]),
                  static_cast<size_t>(default_work_size.data()[1]),
                  static_cast<size_t>(default_work_size.data()[2])};

  VLOG(4) << "out_image: " << out_image;
  VLOG(4) << "global_work_size[3D]: {" << global_work_size[0] << ","
          << global_work_size[1] << "," << global_work_size[2] << "}";

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      cl::NullRange,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);
  context.cl_wait_list()->emplace(out_image, event_);
}

void ConvImageCompute::DepthwiseConv2d3x3s1() {
  const auto& param = *param_.get_mutable<param_t>();
  auto x_dims = param.x->dims();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();
  auto paddings = *param.paddings;
  auto strides = param.strides;
  auto dilations = *param.dilations;

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  auto* input_img = param.x->data<float, cl::Image2D>();
  auto* filter_img = filter_gpu_image_.data<float, cl::Image2D>();

  const cl::Image2D* bias_img = nullptr;
  if (param.bias) {
    bias_img = bias_gpu_image_.data<float, cl::Image2D>();
  }

  auto image_shape = InitImageDimInfoWith(output_dims);

  auto* output_img = param.output->mutable_data<float, cl::Image2D>(
      image_shape["width"], image_shape["height"]);

  STL::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0];
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());

  int c_block = (output_dims[1] + 3) / 4;
  int w = output_dims[3];
  int nh = output_dims[0] * output_dims[2];

  int w_blk_size = 2;
  int w_blk = (w + w_blk_size - 1) / w_blk_size;

  auto global_work_size = cl::NDRange(c_block, w_blk, nh);

  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, static_cast<const int>(c_block));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(w_blk));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(nh));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *input_img);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *filter_img);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *output_img);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(strides[0]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(paddings[0]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(dilations[0]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[1]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[3]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[2]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(output_dims[3]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(output_dims[2]));
  CL_CHECK_FATAL(status);

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      cl::NullRange,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);
  context.cl_wait_list()->emplace(output_img, event_);
}

void ConvImageCompute::DepthwiseConv2d3x3() {
  const auto& param = *param_.get_mutable<param_t>();
  auto x_dims = param.x->dims();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();
  auto paddings = *param.paddings;
  auto strides = param.strides;
  auto dilations = *param.dilations;
  int offset = filter_dims[2] / 2 - paddings[0];
  int input_c_block = (x_dims[1] + 3) / 4;

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  auto* input_img = param.x->data<float, cl::Image2D>();
  auto* filter_img = filter_gpu_image_.data<float, cl::Image2D>();

  const cl::Image2D* bias_img = nullptr;
  if (param.bias) {
    bias_img = bias_gpu_image_.data<float, cl::Image2D>();
  }

  auto image_shape = InitImageDimInfoWith(output_dims);

  auto* output_img = param.output->mutable_data<float, cl::Image2D>(
      image_shape["width"], image_shape["height"]);

  STL::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0];
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());

  int c_block = (output_dims[1] + 3) / 4;
  int w = output_dims[3];
  int nh = output_dims[0] * output_dims[2];
  auto global_work_size = cl::NDRange(c_block, w, nh);

  VLOG(4) << "setArg";
  VLOG(4) << "c_block = " << c_block;
  VLOG(4) << "w = " << w;
  VLOG(4) << "nh = " << nh;

  VLOG(4) << "strides = " << strides[0];
  VLOG(4) << "offset = " << offset;
  VLOG(4) << "dilations = " << dilations[0];
  VLOG(4) << "input_c_block = " << input_c_block;
  VLOG(4) << "x_dims[3] = " << x_dims[3];
  VLOG(4) << "x_dims[2] = " << x_dims[2];
  VLOG(4) << "output_dims[3] = " << output_dims[3];
  VLOG(4) << "output_dims[2] = " << output_dims[2];

  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, static_cast<const int>(c_block));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(w));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(nh));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *input_img);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *filter_img);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *output_img);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(strides[0]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(offset));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(dilations[0]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(input_c_block));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[3]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[2]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(output_dims[3]));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, static_cast<const int>(output_dims[2]));
  CL_CHECK_FATAL(status);

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      cl::NullRange,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);
  context.cl_wait_list()->emplace(output_img, event_);
}

void ConvImageCompute::DepthwiseConv2d() {
  const auto& param = *param_.get_mutable<param_t>();
  auto input_dims = param.x->dims();
  auto paddings = *param.paddings;
  auto strides = param.strides;
  auto* input_image = param.x->data<float, cl::Image2D>();
  auto* filter_image = filter_gpu_image_.data<float, cl::Image2D>();
  auto filter_dims = param.filter->dims();
  auto output_dims = param.output->dims();

  int input_width = input_dims[3];
  int input_height = input_dims[2];
  int output_width = output_dims[3];
  int output_height = output_dims[2];
  int filter_width = filter_dims[3];
  int filter_height = filter_dims[2];
  auto out_image_shape = InitImageDimInfoWith(output_dims);
  auto* out_image = param.output->mutable_data<float, cl::Image2D>(
      out_image_shape["width"], out_image_shape["height"]);

  const bool has_bias = param.bias != nullptr;
  const bool is_element_wise_bias =
      has_bias && param.output->dims() == param.bias->dims();
  int offset = static_cast<int>(param.filter->dims()[2]) / 2 -
               static_cast<int>(paddings[0]);

  // calc input_c_block
  auto input_image_shape = InitImageDimInfoWith(input_dims);
  int input_c_block = input_image_shape["width"] / input_dims[3];
  int input_c = input_dims[1];
  auto dilations = *param.dilations;

  const std::vector<size_t>& default_work_size =
      DefaultWorkSize(output_dims,
                      DDim(std::vector<DDim::value_type>{
                          static_cast<int64_t>(out_image_shape["width"]),
                          static_cast<int64_t>(out_image_shape["height"])}));

  int c_block = default_work_size[0];
  int w = default_work_size[1];
  int nh = default_work_size[2];

  VLOG(4) << "============ depthwise conv2d params ============";
  VLOG(4) << "input_image_shape: " << input_image_shape["width"] << ","
          << input_image_shape["height"];
  VLOG(4) << "input_c_block: " << input_c_block;
  VLOG(4) << "input_c: " << input_c;
  VLOG(4) << "input_image: " << input_image;
  VLOG(4) << "filter_dims: " << filter_dims;
  VLOG(4) << "filter_image: " << filter_image;
  VLOG(4) << "output_dims: " << output_dims;
  VLOG(4) << "out_image_shape: " << out_image_shape["width"] << ", "
          << out_image_shape["height"];
  VLOG(4) << "paddings: " << paddings[0] << "," << paddings[1];
  VLOG(4) << "has bias: " << has_bias;
  VLOG(4) << "is_element_wise_bias : " << is_element_wise_bias;
  VLOG(4) << "strides: " << strides[0] << "," << strides[1];
  VLOG(4) << "offset: " << offset;
  VLOG(4) << "dilations.size : " << dilations.size();
  VLOG(4) << "dilations: " << dilations[0] << ", " << dilations[1];
  VLOG(4) << "default work size{c_block, w, nh}: "
          << "{" << c_block << ", " << w << ", " << nh << ""
          << "}";

  CHECK_GE(dilations.size(), 2);
  CHECK(dilations[0] == dilations[1]);
  CHECK_GE(input_dims.size(), 4);
  CHECK_GE(paddings.size(), 2);
  CHECK(paddings[0] == paddings[1]);
  CHECK_GE(strides.size(), 2);
  CHECK(strides[0] == strides[1]);

  // handle bias  use buffer for channel wise , use image for element wise
  const cl::Buffer* bias_buf = nullptr;
  const cl::Image2D* bias_image = nullptr;
  if (has_bias) {
    bias_image = bias_gpu_image_.data<float, cl::Image2D>();
  }

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  STL::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0];
  auto kernel = context.cl_context()->GetKernel(kernel_key.str());
  VLOG(4) << "kernel_key: " << kernel_key.str();
  VLOG(4) << "kernel ready ... " << kernel_key.str();
  VLOG(4) << "w: " << w;

  cl_int status;
  int arg_idx = 0;
  status = kernel.setArg(arg_idx, c_block);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, w);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, nh);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *input_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, *filter_image);
  CL_CHECK_FATAL(status);
  if (has_bias) {
    VLOG(4) << "set bias_image: ";
    status = kernel.setArg(++arg_idx, *bias_image);
    CL_CHECK_FATAL(status);
  }
  status = kernel.setArg(++arg_idx, *out_image);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, strides[0]);
  CL_CHECK_FATAL(status);

  status = kernel.setArg(++arg_idx, offset);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_c_block);
  CL_CHECK_FATAL(status);

  status = kernel.setArg(++arg_idx, dilations[0]);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, input_height);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, output_height);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, filter_width);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(++arg_idx, filter_height);
  CL_CHECK_FATAL(status);

  auto global_work_size =
      cl::NDRange{static_cast<size_t>(default_work_size.data()[0]),
                  static_cast<size_t>(default_work_size.data()[1]),
                  static_cast<size_t>(default_work_size.data()[2])};

  VLOG(4) << "out_image: " << out_image;
  VLOG(4) << "global_work_size[3D]: {" << global_work_size[0] << ","
          << global_work_size[1] << "," << global_work_size[2] << "}";

  status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      global_work_size,
      cl::NullRange,
      nullptr,
      event_.get());
  CL_CHECK_FATAL(status);
  context.cl_wait_list()->emplace(out_image, event_);
}

void ConvImageCompute::Run() { (this->*impl_)(); }

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// REGISTER_LITE_KERNEL(conv2d,
//                      kOpenCL,
//                      kFloat,
//                      kNCHW,
//                      paddle::lite::kernels::opencl::ConvCompute,
//                      def)
//     .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
//     .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kOpenCL))})
//     .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kOpenCL))})
//     .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kOpenCL))})
//     .Finalize();

REGISTER_LITE_KERNEL(conv2d,
                     kOpenCL,
                     kFloat,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ConvImageCompute,
                     image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d,
                     kOpenCL,
                     kFloat,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ConvImageCompute,
                     image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
