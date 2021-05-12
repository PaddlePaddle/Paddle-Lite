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

#include "lite/kernels/opencl/conv_image_compute.h"

#include <cfloat>
#include <iomanip>
#include <sstream>
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/core/op_registry.h"

#undef LITE_WITH_LOG

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void ConvImageCompute::PrepareForRun() {
  ReInitWhenNeeded();

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  const bool is_mali = context.cl_context()->IsArmMali();
  is_mali_ = context.cl_context()->IsArmMali();
  int compute_units =
      CLRuntime::Global()->GetDeviceInfo()["CL_DEVICE_MAX_COMPUTE_UNITS"];
  auto device_name = CLRuntime::Global()->device().getInfo<CL_DEVICE_NAME>();
  float threshold_2 = FLT_MAX;
  float threshold_4 = FLT_MAX;
  if (device_name.find("Mali-G76") == std::string::npos) {
    threshold_2 = 256.0f * 16.0f;
  } else {
    threshold_2 = 256.0f * 6.0f;
    threshold_4 = 256.0f * 16.0f;
  }
  const bool fp16_support =
      CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
  conv_param_ = param_.get_mutable<param_t>();
  auto output_dims = conv_param_->output->dims();
  output_tensor_n_ = output_dims[0];
  output_tensor_c_ = output_dims[1];
  output_tensor_h_ = output_dims[2];
  output_tensor_w_ = output_dims[3];
  auto task_size = static_cast<float>(output_tensor_h_ * output_tensor_w_ *
                                      output_tensor_c_);
  task_size = task_size / compute_units;
  /*********************************************
   * Initilize attributes
   *********************************************/
  auto filter_dims = conv_param_->filter->dims();
  filter_tensor_n_ = filter_dims[0];
  filter_tensor_c_ = filter_dims[1];
  filter_tensor_h_ = filter_dims[2];
  filter_tensor_w_ = filter_dims[3];

  auto paddings = *conv_param_->paddings;
  pad_up_ = paddings[0];
  pad_down_ = paddings[1];
  pad_left_ = paddings[2];
  pad_right_ = paddings[3];

  auto dilations = *conv_param_->dilations;
  dilation_h_ = dilations[0];
  dilation_w_ = dilations[1];

  stride_h_ = conv_param_->strides[0];
  stride_w_ = conv_param_->strides[1];

  groups_ = conv_param_->groups;
  relu_fused_ = conv_param_->fuse_relu;
  has_bias_ = (conv_param_->bias) != nullptr;
  offset_ = filter_tensor_h_ / 2 - pad_up_;
  offset_w_ = filter_tensor_w_ / 2 - pad_left_;
  offset_h_ = offset_;

  bool pad_equal = ((pad_left_ == pad_up_) && (pad_left_ == pad_right_));
  bool stride_equal = stride_h_ == stride_w_;
  bool dilation_equal = dilation_h_ == dilation_w_;

#ifdef LITE_WITH_LOG
  VLOG(3) << "Is relu fused? / " << (relu_fused_ ? "Yes" : "No");
  VLOG(3) << "groups:" << groups_ << " stride_h_:" << stride_h_
          << " stride_w_:" << stride_w_ << " pad_left_:" << pad_left_
          << " pad_up_:" << pad_up_ << " filter_tensor_h_:" << filter_tensor_h_
          << " filter_tensor_h_:" << filter_tensor_h_;
  VLOG(3) << "input_tensor_nchw:" << input_tensor_n_ << " " << input_tensor_c_
          << " " << input_tensor_h_ << " " << input_tensor_w_;
  VLOG(3) << "dialtion:" << dilation_h_ << " " << dilation_w_;
  VLOG(3) << "output_dims:" << output_tensor_n_ << " " << output_tensor_c_
          << " " << output_tensor_h_ << " " << output_tensor_w_;
  VLOG(3) << "filter_dims:" << filter_tensor_n_ << " " << filter_tensor_c_
          << " " << filter_tensor_h_ << " " << filter_tensor_w_;
  VLOG(3) << "pad_equal:" << pad_equal;
  VLOG(3) << "stride_equal:" << stride_equal;
  VLOG(3) << "dilation_equal:" << dilation_equal;
  VLOG(3) << "padding :" << pad_up_ << " " << pad_down_ << " " << pad_left_
          << " " << pad_right_;
#endif
  CHECK_GE(conv_param_->dilations->size(), 2);
  CHECK_GE(conv_param_->paddings->size(), 2);
  CHECK_GE(conv_param_->strides.size(), 2);

  /*********************************************
   * Upload filter, bias to opencl device
   *********************************************/
  auto* filter_cpu = conv_param_->filter->mutable_data<float>();
  // if (is_mali && filter_tensor_h_ == 1 && filter_tensor_w_ == 1) {
  //   kernel_func_names_.push_back("conv2d_1x1_mali");
  //   kernel_func_paths_.push_back("image/conv2d_1x1_opt_kernel.cl");

  //   auto tensor_hold_filter_buffer = std::unique_ptr<Tensor>(new Tensor);
  //   auto filter_ext_dims = filter_dims;
  //   filter_ext_dims[0] = ROUND_UP(filter_dims[0], 4);
  //   filter_ext_dims[1] = ROUND_UP(filter_dims[1], 4);
  //   tensor_hold_filter_buffer->Resize(filter_ext_dims);
  //   auto* filter_buffer_data =
  //       MUTABLE_DATA_CPU(tensor_hold_filter_buffer.get());
  //   size_t buf_size = tensor_hold_filter_buffer->numel() *
  //                     (fp16_support ? sizeof(half_t) : sizeof(float));
  //   ::memset(filter_buffer_data, 0, buf_size);

  //   OIHW2OHWIO4I4(filter_cpu,
  //                 filter_buffer_data,
  //                 filter_dims[0],
  //                 filter_dims[1],
  //                 filter_dims[2],
  //                 filter_dims[3]);

  //   filter_gpu_buffer_ = std::unique_ptr<Tensor>(new Tensor);

  //   AssignDataFromCPUToGPU(tensor_hold_filter_buffer.get(),
  //                          filter_gpu_buffer_.get());

  //   impl_ = &ConvImageCompute::Conv2d1x1Mali;
  if (is_mali_ && filter_tensor_h_ == 1 && filter_tensor_w_ == 1) {
    filter_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    tensor_hold_filter_image_ = std::unique_ptr<Tensor>(new Tensor);
    tensor_hold_bias_image_ = std::unique_ptr<Tensor>(new Tensor);
    if (task_size <= threshold_2) {
      CLImageConverterNBlock converter;
      kernel_func_names_.push_back("conv2d_1x1_mali_h1w2c1");
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});
      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);

      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      w_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      auto* w_gpu_data = w_gpu_t_->mutable_data(
          TARGET(kOpenCL), tensor_hold_filter_image_->memory_size());
      TargetWrapperCL::MemcpySync(w_gpu_data,
                                  tensor_hold_filter_image_->raw_data(),
                                  tensor_hold_filter_image_->memory_size(),
                                  IoDirection::HtoD);

      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);
    } else if (task_size <= threshold_4) {
      CLImageConverterN2Block converter;
      kernel_func_names_.push_back("conv2d_1x1_mali_h1w2c2");
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});
      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);

      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      w_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      auto* w_gpu_data = w_gpu_t_->mutable_data(
          TARGET(kOpenCL), tensor_hold_filter_image_->memory_size());
      TargetWrapperCL::MemcpySync(w_gpu_data,
                                  tensor_hold_filter_image_->raw_data(),
                                  tensor_hold_filter_image_->memory_size(),
                                  IoDirection::HtoD);

      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);
    } else {
      CLImageConverterN2Block converter;
      kernel_func_names_.push_back("conv2d_1x1_mali_h2w2c2");
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});
      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);
      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      w_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      auto* w_gpu_data = w_gpu_t_->mutable_data(
          TARGET(kOpenCL), tensor_hold_filter_image_->memory_size());
      TargetWrapperCL::MemcpySync(w_gpu_data,
                                  tensor_hold_filter_image_->raw_data(),
                                  tensor_hold_filter_image_->memory_size(),
                                  IoDirection::HtoD);

      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);
    }
    kernel_func_paths_.push_back("image/conv2d_1x1_opt_kernel.cl");
    impl_ = &ConvImageCompute::Conv2d1x1opt;
  } else {
    filter_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    tensor_hold_filter_image_ = std::unique_ptr<Tensor>(new Tensor);
    tensor_hold_bias_image_ = std::unique_ptr<Tensor>(new Tensor);

    if (filter_tensor_h_ == 1 && filter_tensor_w_ == 1) {
      CHECK(pad_equal && stride_equal && dilation_equal);
      if (input_tensor_c_ % 4 == 0) {
        kernel_func_names_.push_back("conv2d_1x1_h1w4c1");
      } else {
        kernel_func_names_.push_back("conv2d_1x1_opt");
      }
      kernel_func_paths_.push_back("image/conv2d_1x1_opt_kernel.cl");

      CLImageConverterNWBlock converter;
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});
      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);

      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);

      impl_ = &ConvImageCompute::Conv2d1x1opt;
#define DEPTH_CONV_USE_SPL
#ifdef DEPTH_CONV_USE_SPL
    } else if (filter_tensor_c_ == 1 && input_tensor_c_ == output_tensor_c_ &&
               filter_tensor_h_ == 3 && filter_tensor_w_ == 3 && groups_ > 1) {
      // depth_conv2d_3x3s1, depth_conv2d_3x3
      CHECK(dilation_equal);
      if (stride_equal && stride_h_ == 1 && dilation_h_ == 1) {
        kernel_func_names_.push_back("depth_conv2d_3x3s1");
        impl_ = &ConvImageCompute::DepthwiseConv2d3x3s1;
      } else {
        kernel_func_names_.push_back("depth_conv2d_3x3");
        impl_ = &ConvImageCompute::DepthwiseConv2d3x3;
      }
      kernel_func_paths_.push_back("image/depthwise_conv2d_kernel.cl");

      CLImageConverterNWBlock converter;
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});

      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);
      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);
#endif
    } else if (filter_tensor_c_ == 1 && input_tensor_c_ == output_tensor_c_
#ifdef DEPTH_CONV_USE_SPL
               &&
               filter_tensor_h_ != 3
#endif
#undef DEPTH_CONV_USE_SPL
               ) {
      // common depth_conv2d
      kernel_func_names_.push_back("depth_conv2d");
      kernel_func_paths_.push_back("image/depthwise_conv2d_basic_kernel.cl");

      CLImageConverterNWBlock converter;
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});

      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);
      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);

      impl_ = &ConvImageCompute::DepthwiseConv2d;
    } else if (filter_tensor_h_ == 3 && filter_tensor_w_ == 3 &&
               dilation_h_ == 1 && dilation_w_ == 1) {
      // conv2d_3x3
      pad_equal = (pad_left_ == pad_up_);
      CHECK(pad_equal && stride_equal && dilation_equal);
      if (groups_ == 1) {
        kernel_func_names_.push_back(
            input_tensor_n_ > 1 ? "conv2d_3x3_multi_batch" : "conv2d_3x3_opt");
        kernel_func_paths_.push_back("image/conv2d_3x3_opt_kernel.cl");
        impl_ = &ConvImageCompute::Conv2d3x3opt;

        CLImageConverterNBlock converter;
        const DDim& filter_image_dims =
            converter.InitImageDimInfoWith(filter_dims);
        filter_image_h_ = filter_image_dims[1];
        filter_image_w_ = filter_image_dims[0];
        tensor_hold_filter_image_->Resize(
            {1, filter_image_w_, filter_image_h_, 4});
        auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);

        converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
        MUTABLE_DATA_GPU(filter_gpu_image_,
                         filter_image_w_,
                         filter_image_h_,
                         filter_image_data);
      } else {  // groups_ > 1
        kernel_func_names_.push_back("conv2d_3x3");
        kernel_func_paths_.push_back("image/conv2d_3x3_kernel.cl");
        impl_ = &ConvImageCompute::Conv2d3x3;

        CLImageConverterFolder converter;
        const DDim& filter_image_dims =
            converter.InitImageDimInfoWith(filter_dims);
        filter_image_h_ = filter_image_dims[1];
        filter_image_w_ = filter_image_dims[0];
        tensor_hold_filter_image_->Resize(
            {1, filter_image_w_, filter_image_h_, 4});
        auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);

        converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
        MUTABLE_DATA_GPU(filter_gpu_image_,
                         filter_image_w_,
                         filter_image_h_,
                         filter_image_data);
      }
    } else if (filter_tensor_h_ == 5 && filter_tensor_w_ == 5 && pad_equal &&
               stride_equal && dilation_equal) {
#define CONV_5x5_OPT
#ifndef CONV_5x5_OPT
      // conv2d_5x5
      kernel_func_names_.push_back("conv2d_5x5");
      kernel_func_paths_.push_back("image/conv2d_5x5_kernel.cl");

      CLImageConverterFolder converter;
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});

      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);
      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);

      impl_ = &ConvImageCompute::Conv2d5x5;
#else
      // conv2d_5x5_opt

      kernel_func_names_.push_back(
          input_tensor_n_ > 1 ? "conv2d_5x5_multi_batch" : "conv2d_5x5_opt");
      kernel_func_paths_.push_back("image/conv2d_5x5_opt_kernel.cl");

      CLImageConverterFolder converter;
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});

      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);

      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);

      impl_ = &ConvImageCompute::Conv2d5x5opt;
#endif
#undef CONV_5x5_OPT
    } else if (filter_tensor_h_ == 7 && filter_tensor_w_ == 7 && pad_equal &&
               stride_equal && dilation_equal) {
#define CONV_7x7_OPT
#ifndef CONV_7x7_OPT
      // conv2d_7x7
      kernel_func_names_.push_back("conv2d_7x7");
      kernel_func_paths_.push_back("image/conv2d_7x7_kernel.cl");

      CLImageConverterFolder converter;
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});

      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);
      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);
      impl_ = &ConvImageCompute::Conv2d7x7;

#else
      // conv2d_7x7
      kernel_func_names_.push_back(
          input_tensor_n_ > 1 ? "conv2d_7x7_multi_batch" : "conv2d_7x7_opt");
      kernel_func_paths_.push_back("image/conv2d_7x7_opt_kernel.cl");

      CLImageConverterFolder converter;
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});

      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);
      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);

      impl_ = &ConvImageCompute::Conv2d7x7opt;
#endif
#undef CONV_7x7_OPT
    } else if (groups_ == 1) {
      // conv2d_common
      kernel_func_names_.push_back("conv2d_common");
      kernel_func_paths_.push_back("image/conv2d_common_kernel.cl");
      impl_ = &ConvImageCompute::Conv2dCommon;

      CLImageConverterNBlock converter;
      const DDim& filter_image_dims =
          converter.InitImageDimInfoWith(filter_dims);
      filter_image_h_ = filter_image_dims[1];
      filter_image_w_ = filter_image_dims[0];
      tensor_hold_filter_image_->Resize(
          {1, filter_image_w_, filter_image_h_, 4});
      auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);

      converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
      MUTABLE_DATA_GPU(filter_gpu_image_,
                       filter_image_w_,
                       filter_image_h_,
                       filter_image_data);

    } else {
      LOG(FATAL) << "conv image compute not support this condition yet! ";
    }
  }  // if (is_mali)
  VLOG(1) << "kernel_func_names_[0]:" << kernel_func_names_[0]
          << " kernel_func_paths_[0]:" << kernel_func_paths_[0];

  // build options
  std::string build_options_single{""};
  // relu options
  VLOG(3) << "relu_fused_:" << relu_fused_
          << " conv_param_->activation_param.active_type:"
          << static_cast<int>(conv_param_->activation_param.active_type)
          << " conv_param_->activation_param.has_active:"
          << conv_param_->activation_param.has_active;
  // alpha_image_p_ init
  alpha_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
  std::unique_ptr<Tensor> tensor_hold_alpha_image =
      std::unique_ptr<Tensor>(new Tensor);
  tensor_hold_alpha_image->Resize({1, 1, 1, 4});
  auto* alpha_image_data = DATA_GPU(tensor_hold_alpha_image);
  MUTABLE_DATA_GPU(alpha_gpu_image_, 1, 1, alpha_image_data);
  alpha_image_p_ = DATA_GPU(alpha_gpu_image_);
  if (conv_param_->activation_param.has_active) {
    if (conv_param_->activation_param.active_type ==
        lite_api::ActivationType::kRelu) {
      build_options_single += " -DRELU";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kRelu6) {
      build_options_single += " -DRELU6";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kLeakyRelu) {
      std::string leaky_relu_alpha_str =
          std::to_string(conv_param_->activation_param.Leaky_relu_alpha);
      build_options_single +=
          " -DLEAKY_RELU -DLEAKY_RELU_ALPHA=" + leaky_relu_alpha_str + "f";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kHardSwish) {
      std::string threshold =
          std::to_string(conv_param_->activation_param.hard_swish_threshold);
      std::string scale =
          std::to_string(conv_param_->activation_param.hard_swish_scale);
      std::string offset =
          std::to_string(conv_param_->activation_param.hard_swish_offset);
      build_options_single += " -DHARD_SWISH -DACT_THRESHOLD=" + threshold +
                              "f" + " -DACT_SCALE=" + scale + "f" +
                              " -DACT_OFFSET=" + offset + "f";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kHardSigmoid) {
      std::string slope =
          std::to_string(conv_param_->activation_param.hard_sigmoid_slope);
      std::string offset =
          std::to_string(conv_param_->activation_param.hard_sigmoid_offset);
      build_options_single += " -DHARD_SIGMOID -DHARD_SIGMOID_SLOPE=" + slope +
                              "f" + " -DHARD_SIGMOID_OFFSET=" + offset + "f";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kPRelu) {
      std::string prelu_mode = conv_param_->activation_param.Prelu_mode;
      build_options_single += " -DPRELU";
      if (prelu_mode == "channel") {
        build_options_single += " -DPRELU_CH";
      } else if (prelu_mode == "element") {
        build_options_single += " -DPRELU_ELE";
      } else {
        build_options_single += " -DPRELU_ALL";
      }
      CLImageConverterFolder alpha_converter;
      const DDim& alpha_image_dims = alpha_converter.InitImageDimInfoWith(
          conv_param_->activation_param.Prelu_alpha->dims());
      tensor_hold_alpha_image->Resize(
          {1, alpha_image_dims[0], alpha_image_dims[1], 4});
      auto* alpha_image_data = MUTABLE_DATA_CPU(tensor_hold_alpha_image);
      auto* alpha_cpu_data =
          conv_param_->activation_param.Prelu_alpha->mutable_data<float>();
      alpha_converter.NCHWToImage(
          alpha_cpu_data,
          alpha_image_data,
          conv_param_->activation_param.Prelu_alpha->dims());
      MUTABLE_DATA_GPU(alpha_gpu_image_,
                       alpha_image_dims[0],
                       alpha_image_dims[1],
                       alpha_image_data);
      alpha_image_p_ = DATA_GPU(alpha_gpu_image_);
    } else {
      LOG(FATAL) << "Unsupported activation type:"
                 << static_cast<int>(conv_param_->activation_param.active_type);
    }
  }
  SetGlobalWorkSize();

  // bias options
  const bool is_element_wise_bias =
      has_bias_ && conv_param_->output->dims() == conv_param_->bias->dims();
  if (has_bias_) {
    build_options_single +=
        is_element_wise_bias ? " -DBIASE_ELE" : " -DBIASE_CH";
  }

  // convert cpu buffer bias --> gpu buffer/image
  // if (has_bias_ && is_mali && filter_gpu_buffer_ != nullptr) {
  //   bias_gpu_buffer_ = std::unique_ptr<Tensor>(new Tensor);
  //   auto bias_dims = conv_param_->bias->dims();
  //   auto tensor_hold_bias_buffer = std::unique_ptr<Tensor>(new Tensor);
  //   auto bias_ext_dims = bias_dims;
  //   bias_ext_dims[0] = ROUND_UP(bias_dims[0], 4);
  //   tensor_hold_bias_buffer->Resize(bias_ext_dims);
  //   auto* bias_buffer_data = MUTABLE_DATA_CPU(tensor_hold_bias_buffer.get());
  //   size_t buf_size = tensor_hold_bias_buffer->numel() *
  //                     (fp16_support ? sizeof(half_t) : sizeof(float));
  //   ::memset(bias_buffer_data, 0, buf_size);

  //   float* bias_fp32 = static_cast<float*>(bias_buffer_data);
  //   half_t* bias_fp16 = static_cast<half_t*>(bias_buffer_data);
  //   for (auto i = 0; i < bias_dims.production(); ++i) {
  //     fp16_support
  //         ? bias_fp16[i] =
  //               Float2Half(conv_param_->bias->mutable_data<float>()[i])
  //         : bias_fp32[i] = conv_param_->bias->mutable_data<float>()[i];
  //   }

  //   AssignDataFromCPUToGPU(tensor_hold_bias_buffer.get(),
  //                          bias_gpu_buffer_.get());
  // } else if (is_mali && filter_gpu_buffer_ != nullptr) {
  //   bias_gpu_buffer_ = std::unique_ptr<Tensor>(new Tensor);
  //   auto tensor_hold_bias_buffer = std::unique_ptr<Tensor>(new Tensor);
  //   DDimLite bias_ext_dims({4});
  //   tensor_hold_bias_buffer->Resize(bias_ext_dims);
  //   auto* bias_buffer_data = MUTABLE_DATA_CPU(tensor_hold_bias_buffer.get());
  //   AssignDataFromCPUToGPU(tensor_hold_bias_buffer.get(),
  //                          bias_gpu_buffer_.get());
  // } else if (has_bias_) {
  if (has_bias_) {
    bias_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    CLImageConverterFolder bias_converter;
    const DDim& bias_image_dims =
        bias_converter.InitImageDimInfoWith(conv_param_->bias->dims());
    bias_image_h_ = bias_image_dims[1];
    bias_image_w_ = bias_image_dims[0];
    tensor_hold_bias_image_->Resize(
        {1, bias_image_dims[0], bias_image_dims[1], 4});

    auto* bias_image_data = MUTABLE_DATA_CPU(tensor_hold_bias_image_);
    auto* bias_cpu_data = conv_param_->bias->mutable_data<float>();
    bias_converter.NCHWToImage(
        bias_cpu_data, bias_image_data, conv_param_->bias->dims());

    MUTABLE_DATA_GPU(bias_gpu_image_,
                     bias_image_dims[0],
                     bias_image_dims[1],
                     bias_image_data);
  } else {
    bias_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    CLImageConverterFolder bias_converter;
    tensor_hold_bias_image_->Resize({1, 1, 1, 4});
    auto* bias_image_data = DATA_GPU(tensor_hold_bias_image_);
    MUTABLE_DATA_GPU(bias_gpu_image_, 1, 1, bias_image_data);
  }

  // scale options
  if (conv_param_->scale_activation_type == "") {
    // do nothing
  } else if (conv_param_->scale_activation_type == "relu6") {
    build_options_single += " -DSCALE_ACTIVATION -DFUSE_SCALE_RELU6 ";
  } else {
    LOG(FATAL) << "Unsupported scale_activation_type:"
               << conv_param_->scale_activation_type;
  }

  // define buffer/image pointer for filter & bias
  // if (is_mali && filter_tensor_h_ == 1 && filter_tensor_w_ == 1) {
  //   fp16_support
  //       ? filter_buffer_p_ =
  //             filter_gpu_buffer_->mutable_data<half_t, cl::Buffer>()
  //       : filter_buffer_p_ =
  //             filter_gpu_buffer_->mutable_data<float, cl::Buffer>();
  //   if (has_bias_) {
  //     fp16_support
  //         ? bias_buffer_p_ =
  //               bias_gpu_buffer_->mutable_data<half_t, cl::Buffer>()
  //         : bias_buffer_p_ =
  //               bias_gpu_buffer_->mutable_data<float, cl::Buffer>();
  //   }
  // } else {
  filter_image_p_ = DATA_GPU(filter_gpu_image_);
  bias_image_p_ = DATA_GPU(bias_gpu_image_);

  build_options_.push_back(build_options_single);
  for (size_t i = 0; i < kernel_func_names_.size(); i++) {
    context.cl_context()->AddKernel(kernel_func_names_[i],
                                    kernel_func_paths_[i],
                                    build_options_[i],
                                    time_stamp_);
  }
  SetLocalWorkSize(CLRuntime::Global()->lws_repeats());
}

#define SHOW_EACH_LWS_TIME
#undef SHOW_EACH_LWS_TIME
void ConvImageCompute::SetLocalWorkSize(size_t repeats /*=4*/) {
  if (kernel_func_names_[0] == "conv2d_1x1_h1w4c1") {
    auto tuned_map_key = GenerateTunedKey();
    cl::NDRange lws_in_map = cl::NullRange;
    // if (CLRuntime::Global()->HasTunedLocalWorkSizeMap(tuned_map_key,
    //                                                   &lws_in_map)) {
    //   local_work_size_ = lws_in_map;
    //   return;
    // }
    std::string final_kernel_func_name = "conv2d_1x1_h1w4c1";
    cl::NDRange final_global_work_size = cl::NDRange{
        static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
    cl::NDRange final_local_work_size = cl::NDRange{
        static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
    double final_lws_time = DBL_MAX;
    auto& context = ctx_->As<OpenCLContext>();
    std::stringstream kernel_key;
    for (size_t i = 0; i < 4; i++) {
      if (i == 1) {
        kernel_func_names_[0] = "conv2d_1x1_h1w5c1";
        global_work_size_ =
            cl::NDRange{static_cast<size_t>(default_c_blk_),
                        static_cast<size_t>(UP_DIV(default_w_blk_, 5)),
                        static_cast<size_t>(default_nh_blk_)};
        context.cl_context()->AddKernel(kernel_func_names_[0],
                                        kernel_func_paths_[0],
                                        build_options_[0],
                                        time_stamp_);
      }
      if (i == 2) {
        kernel_func_names_[0] = "conv2d_1x1_h1w7c1";
        global_work_size_ =
            cl::NDRange{static_cast<size_t>(default_c_blk_),
                        static_cast<size_t>(UP_DIV(default_w_blk_, 7)),
                        static_cast<size_t>(default_nh_blk_)};
        context.cl_context()->AddKernel(kernel_func_names_[0],
                                        kernel_func_paths_[0],
                                        build_options_[0],
                                        time_stamp_);
      }
      if (i == 3) {
        kernel_func_names_[0] = "conv2d_1x1_h2w2c2";
        global_work_size_ =
            cl::NDRange{static_cast<size_t>(UP_DIV(default_c_blk_, 2)),
                        static_cast<size_t>(UP_DIV(default_w_blk_, 2)),
                        static_cast<size_t>(UP_DIV(default_nh_blk_, 2))};
        context.cl_context()->AddKernel(kernel_func_names_[0],
                                        kernel_func_paths_[0],
                                        build_options_[0],
                                        time_stamp_);
      }
      kernel_key.str("");
      kernel_key << kernel_func_names_[0] << build_options_[0] << time_stamp_;
      kernel_ = context.cl_context()->GetKernel(kernel_key.str());

      size_t max_work_group_size = 0;
      kernel_.getWorkGroupInfo<size_t>(CLRuntime::Global()->device(),
                                       CL_KERNEL_WORK_GROUP_SIZE,
                                       &max_work_group_size);
      std::set<cl::NDRange> lwss = context.cl_context()->GenerateLocalWorkSizes(
          global_work_size_, max_work_group_size);
      CHECK(lwss.size() > 0)
          << "Possible local work sizes should bigger than zero";
      local_work_size_ = *lwss.begin();
      if (max_work_group_size <= 0 || !use_lws_ ||
          CLRuntime::Global()->auto_tune() <= 0) {
        if (!use_lws_) {
          local_work_size_ = cl::NullRange;
        }
        return;
      }
      double min_lws_time = DBL_MAX;
      cl::NDRange min_lws = *lwss.begin();
      for (cl::NDRange cur_lws : lwss) {
        local_work_size_ = cur_lws;
        double cur_lws_time = 0.0f;
        // note: useless for skip first run
        for (size_t i = 0; i < repeats; ++i) {
          Run();
          cur_lws_time += CLRuntime::Global()->GetCommandTime(event_);
        }
        cur_lws_time /= repeats;
        if (min_lws_time > cur_lws_time) {
          min_lws = cur_lws;
          min_lws_time = cur_lws_time;
        }
      }
      if (final_lws_time > min_lws_time) {
        final_kernel_func_name = kernel_func_names_[0];
        final_global_work_size = global_work_size_;
        final_local_work_size = min_lws;
        final_lws_time = min_lws_time;
      }
    }
    kernel_func_names_[0] = final_kernel_func_name;
    global_work_size_ = final_global_work_size;
    local_work_size_ = final_local_work_size;
    kernel_key.str("");
    kernel_key << kernel_func_names_[0] << build_options_[0] << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
    if (kernel_func_names_[0] == "conv2d_1x1_h1w5c1") {
      w_blk_ = UP_DIV(default_w_blk_, 5);
    }
    if (kernel_func_names_[0] == "conv2d_1x1_h1w7c1") {
      w_blk_ = UP_DIV(default_w_blk_, 7);
    }
    // CLRuntime::Global()->SetTunedLocalWorkSizeMap(tuned_map_key,local_work_size_);
  } else {
    auto& context = ctx_->As<OpenCLContext>();
    std::stringstream kernel_key;
    kernel_key << kernel_func_names_[0] << build_options_[0] << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());

    auto tuned_map_key = GenerateTunedKey();
    cl::NDRange lws_in_map = cl::NullRange;
    if (CLRuntime::Global()->HasTunedLocalWorkSizeMap(tuned_map_key,
                                                      &lws_in_map)) {
      local_work_size_ = lws_in_map;
      return;
    }

    size_t max_work_group_size = 0;
    kernel_.getWorkGroupInfo<size_t>(CLRuntime::Global()->device(),
                                     CL_KERNEL_WORK_GROUP_SIZE,
                                     &max_work_group_size);
    std::set<cl::NDRange> lwss = context.cl_context()->GenerateLocalWorkSizes(
        global_work_size_, max_work_group_size);
    CHECK(lwss.size() > 0)
        << "Possible local work sizes should bigger than zero";
    local_work_size_ = *lwss.begin();
    if (max_work_group_size <= 0 || !use_lws_ ||
        CLRuntime::Global()->auto_tune() <= 0) {
      if (!use_lws_) {
        local_work_size_ = cl::NullRange;
      }
      return;
    }

#ifdef SHOW_EACH_LWS_TIME
    LOG(INFO) << "====== start =======";
#endif
    double min_lws_time = DBL_MAX;
    cl::NDRange min_lws = *lwss.begin();
    for (cl::NDRange cur_lws : lwss) {
      local_work_size_ = cur_lws;
      double cur_lws_time = 0.0f;
      // note: useless for skip first run
      for (size_t i = 0; i < repeats; ++i) {
        Run();
        cur_lws_time += CLRuntime::Global()->GetCommandTime(event_);
      }
      cur_lws_time /= repeats;
#ifdef SHOW_EACH_LWS_TIME
      LOG(INFO) << GenerateTunedKey() << " "
                << "{" << std::to_string(local_work_size_[0]) << ","
                << std::to_string(local_work_size_[1]) << ","
                << std::to_string(local_work_size_[2]) << "} -->"
                << cur_lws_time;
#endif
      if (min_lws_time > cur_lws_time) {
        min_lws = cur_lws;
        min_lws_time = cur_lws_time;
      }
    }
#ifdef SHOW_EACH_LWS_TIME
    LOG(INFO) << "=======================";
    LOG(INFO) << "best:" << std::to_string(min_lws[0]) << ","
              << std::to_string(min_lws[1]) << "," << std::to_string(min_lws[2])
              << ","
              << " time:" << min_lws_time;
    LOG(INFO) << "======= finish ========";
#endif
    local_work_size_ = min_lws;
    CLRuntime::Global()->SetTunedLocalWorkSizeMap(tuned_map_key,
                                                  local_work_size_);
  }
}

std::string ConvImageCompute::GenerateTunedKey() {
  std::stringstream key;
  key << kernel_func_names_[0] << ",x:" << input_tensor_n_ << "x"
      << input_tensor_c_ << "x" << input_tensor_h_ << "x" << input_tensor_w_
      << ",w:" << filter_tensor_n_ << "x" << filter_tensor_c_ << "x"
      << filter_tensor_h_ << "x" << filter_tensor_w_ << ",b:" << bias_image_h_
      << "x" << bias_image_w_ << ",pad:" << pad_up_ << pad_down_ << pad_left_
      << pad_right_ << ",dil:" << dilation_h_ << dilation_w_
      << ",s:" << stride_h_ << stride_w_ << ",g:" << groups_
      << ",act:" << static_cast<int>(conv_param_->activation_param.active_type);
  return key.str();
}

void ConvImageCompute::ReInitWhenNeeded() {
  conv_param_ = param_.get_mutable<param_t>();
  auto x_dims = conv_param_->x->dims();
#ifdef LITE_WITH_LOG
  LOG(INFO) << "is_first_epoch_for_run_:" << is_first_epoch_for_run_
            << ", last_input_dims_:" << last_input_dims_
            << ", x_dims:" << x_dims;
#endif

  if (is_first_epoch_for_run_ || last_input_dims_ != x_dims) {
    is_first_epoch_for_run_ = false;
    last_input_dims_ = x_dims;

    input_tensor_n_ = x_dims[0];
    input_tensor_c_ = x_dims[1];
    input_tensor_h_ = x_dims[2];
    input_tensor_w_ = x_dims[3];
    auto x_image_shape = InitImageDimInfoWith(x_dims);
    input_image_h_ = x_image_shape["height"];
    input_image_w_ = x_image_shape["width"];

    auto output_dims = conv_param_->output->dims();
    output_tensor_n_ = output_dims[0];
    output_tensor_c_ = output_dims[1];
    output_tensor_h_ = output_dims[2];
    output_tensor_w_ = output_dims[3];
    auto output_image_shape = InitImageDimInfoWith(output_dims);
    output_image_h_ = output_image_shape["height"];
    output_image_w_ = output_image_shape["width"];

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    CHECK_GE(conv_param_->x->dims().size(), 4);
    CHECK_GE(conv_param_->output->dims().size(), 4);

    // define image pointer for input, output
    input_image_p_ = DATA_GPU(conv_param_->x);
    output_image_p_ = MUTABLE_DATA_GPU(
        conv_param_->output, output_image_w_, output_image_h_, nullptr);

    SetGlobalWorkSize();
  }
}

void ConvImageCompute::SetGlobalWorkSize() {
  if (kernel_func_names_.size() <= 0) return;
  // general input_c_block
  input_c_block_ = static_cast<int>(input_image_w_ / input_tensor_w_);

  // general gws
  auto output_dims = conv_param_->output->dims();
  const std::vector<size_t>& default_work_size =
      DefaultGlobalWorkSize(output_dims,
                            DDim(std::vector<DDim::value_type>{
                                static_cast<int64_t>(output_image_w_),
                                static_cast<int64_t>(output_image_h_)}));
  default_c_blk_ = default_work_size[0];
  default_w_blk_ = default_work_size[1];
  default_nh_blk_ = default_work_size[2];
  c_blk_ = default_c_blk_;
  w_blk_ = default_w_blk_;
  nh_blk_ = default_nh_blk_;
  global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                  static_cast<size_t>(w_blk_),
                                  static_cast<size_t>(nh_blk_)};

  if (kernel_func_names_[0] == "conv2d_1x1_mali") {
    global_work_size_ =
        cl::NDRange{static_cast<size_t>(c_blk_ * UP_DIV(w_blk_, 4)),
                    static_cast<size_t>(nh_blk_)};

  } else if (kernel_func_names_[0] == "conv2d_1x1_h1w4c1" ||
             kernel_func_names_[0] == "conv2d_1x1_opt") {
    w_blk_ = UP_DIV(default_w_blk_, 4);
    c_blk_ = default_c_blk_;
    nh_blk_ = default_nh_blk_;
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};
  } else if (kernel_func_names_[0] == "conv2d_1x1_mali_h1w2c1") {
    w_blk_ = maptofactor(default_w_blk_, 2);
    c_blk_ = default_c_blk_;
    nh_blk_ = default_nh_blk_;
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};
  } else if (kernel_func_names_[0] == "conv2d_1x1_mali_h1w2c2") {
    w_blk_ = maptofactor(default_w_blk_, 2);
    c_blk_ = maptofactor(default_c_blk_, 2);
    nh_blk_ = default_nh_blk_;
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};
  } else if (kernel_func_names_[0] == "conv2d_1x1_mali_h2w2c2") {
    w_blk_ = maptofactor(default_w_blk_, 2);
    c_blk_ = maptofactor(default_c_blk_, 2);
    nh_blk_ = maptofactor(default_nh_blk_, 2);
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};

  } else if (kernel_func_names_[0] == "depth_conv2d_3x3s1") {
    // depthwise spl gws s1
    int c_block = (output_tensor_c_ + 3) / 4;
    int w = output_tensor_w_;
    int nh = output_tensor_n_ * output_tensor_h_;
    int w_blk_size = 2;
    int w_blk = (w + w_blk_size - 1) / w_blk_size;

    c_blk_ = c_block;
    w_blk_ = w_blk;
    nh_blk_ = nh;
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};
  } else if (kernel_func_names_[0] == "depth_conv2d_3x3") {
    // depthwise spl gws
    int c_block = (output_tensor_c_ + 3) / 4;
    int w = output_tensor_w_;
    int nh = output_tensor_n_ * output_tensor_h_;

    c_blk_ = c_block;
    w_blk_ = w;
    nh_blk_ = nh;
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};
    input_c_block_ = static_cast<const int>((input_tensor_c_ + 3) / 4);
  } else if (kernel_func_names_[0] == "depth_conv2d_common") {
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>((w_blk_ + 3) / 4),
                                    static_cast<size_t>(nh_blk_)};
  } else if (kernel_func_names_[0] == "conv2d_3x3") {
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};

  } else if (kernel_func_names_[0] == "conv2d_3x3_multi_batch" ||
             kernel_func_names_[0] == "conv2d_3x3_opt") {
    int w_blk_size = 5;
    int w_blk = (default_w_blk_ + w_blk_size - 1) / w_blk_size;

    int h_blk_size = 1;
    int h_blk = (default_nh_blk_ + h_blk_size - 1) / h_blk_size;

    c_blk_ = default_c_blk_;
    w_blk_ = w_blk;
    nh_blk_ = h_blk;

    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};
  } else if (kernel_func_names_[0] == "conv2d_5x5_multi_batch" ||
             kernel_func_names_[0] == "conv2d_5x5_opt") {
    int w_blk_size = 5;
    int w_blk = (default_w_blk_ + w_blk_size - 1) / w_blk_size;

    int h_blk_size = 1;
    int h_blk = (default_nh_blk_ + h_blk_size - 1) / h_blk_size;

    c_blk_ = default_c_blk_;
    w_blk_ = w_blk;
    nh_blk_ = h_blk;
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};
  } else if (kernel_func_names_[0] == "conv2d_7x7_multi_batch" ||
             kernel_func_names_[0] == "conv2d_7x7_opt") {
    int w_blk_size = 5;
    int w_blk = (default_w_blk_ + w_blk_size - 1) / w_blk_size;

    int h_blk_size = 1;
    int h_blk = (default_nh_blk_ + h_blk_size - 1) / h_blk_size;

    c_blk_ = default_c_blk_;
    w_blk_ = w_blk;
    nh_blk_ = h_blk;
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};
  } else if (kernel_func_names_[0] == "conv2d_common") {
    c_blk_ = (output_tensor_c_ + 3) / 4;
    w_blk_ = maptofactor(default_w_blk_, 4);
    nh_blk_ = default_nh_blk_;
    global_work_size_ = cl::NDRange{static_cast<size_t>(c_blk_),
                                    static_cast<size_t>(w_blk_),
                                    static_cast<size_t>(nh_blk_)};
    input_c_block_ = static_cast<const int>((input_tensor_c_ + 3) / 4);
  }
  VLOG(4) << "global_work_size_[3D]: {" << global_work_size_[0] << ","
          << global_work_size_[1] << "," << global_work_size_[2] << "}";
  VLOG(4) << "local_work_size_[3D]: {" << local_work_size_[0] << ","
          << local_work_size_[1] << "," << local_work_size_[2] << "}";
  for (auto i = 0; i < global_work_size_.dimensions(); i++) {
    VLOG(4) << "global_work_size[" << i << "]: " << global_work_size_[i];
  }
}

void ConvImageCompute::OIHW2OHWIO4I4(
    void* src, void* dst, size_t O, size_t I, size_t H, size_t W) {
  bool fp16_support =
      CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
  size_t i_block = UP_DIV(I, 4);

  float* dst_fp32 = static_cast<float*>(dst);
  half_t* dst_fp16 = static_cast<half_t*>(dst);

  float* p = static_cast<float*>(src);
  for (size_t o = 0; o < O; o++) {
    int o_idx = o / 4;
    for (size_t i = 0; i < I; i++) {
      int i_idx = i / 4;
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          size_t dst_idx = o_idx * H * W * i_block * 4 * 4 +
                           h * W * i_block * 4 * 4 + w * i_block * 4 * 4 +
                           i_idx * 4 * 4 + (o % 4) * 4 + (i % 4);
          fp16_support ? dst_fp16[dst_idx] = Float2Half(*p)
                       : dst_fp32[dst_idx] = *p;
          p++;
        }
      }
    }
  }
}

void ConvImageCompute::AssignDataFromCPUToGPU(const Tensor* tensor_cpu_p,
                                              Tensor* tensor_gpu_p) {
  bool fp16_support =
      lite::CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
  fp16_support
      ? tensor_gpu_p->Assign<half_t, lite::DDim, TARGET(kOpenCL)>(
            tensor_cpu_p->data<half_t>(), tensor_cpu_p->dims())
      : tensor_gpu_p->Assign<float, lite::DDim, TARGET(kOpenCL)>(
            tensor_cpu_p->data<float>(), tensor_cpu_p->dims());
}

void ConvImageCompute::Conv2d1x1Mali() {
  cl_int4 input_shape = {input_tensor_n_,
                         input_tensor_h_,
                         input_tensor_w_,
                         UP_DIV(input_tensor_c_, 4)};
  cl_int4 output_shape = {output_tensor_n_,
                          output_tensor_h_,
                          output_tensor_w_,
                          UP_DIV(output_tensor_c_, 4)};
  cl_int2 stride = {stride_h_, stride_w_};
  cl_int4 pad = {pad_up_, pad_down_, pad_left_, pad_right_};

  int cnt = 0;
  status_ = kernel_.setArg(cnt++, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(cnt++, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(cnt++, *filter_buffer_p_);
  CL_CHECK_FATAL(status_);
  if (has_bias_) {
    status_ = kernel_.setArg(cnt++, *bias_buffer_p_);
    CL_CHECK_FATAL(status_);
  }
  status_ = kernel_.setArg(cnt++, input_shape);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(cnt++, output_shape);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(cnt++, stride);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(cnt++, pad);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(cnt++, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Conv2d1x1opt() {
  status_ = kernel_.setArg(0, default_c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  if (is_mali_) {
    auto* filter_buffer_p_ = w_gpu_t_->data<half_t, cl::Buffer>();
    status_ = kernel_.setArg(4, *filter_buffer_p_);
  } else {
    status_ = kernel_.setArg(4, *filter_image_p_);
  }
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, offset_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, input_c_block_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, input_tensor_c_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(16, default_w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(17, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Conv2d3x3() {
  use_lws_ = false;
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, offset_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, input_c_block_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, output_tensor_c_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(16, filter_tensor_c_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(17, filter_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(18, filter_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(19, groups_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(20, input_tensor_c_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(21, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Conv2d3x3opt() {
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, pad_left_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, input_tensor_n_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, input_tensor_c_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(16, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Conv2d5x5() {
  use_lws_ = false;
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, offset_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, input_c_block_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Conv2d5x5opt() {
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, pad_left_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, input_tensor_n_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, input_tensor_c_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(16, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Conv2d7x7() {
  use_lws_ = false;
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, offset_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, input_c_block_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Conv2d7x7opt() {
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, pad_left_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, input_tensor_n_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, input_tensor_c_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(16, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::DepthwiseConv2d3x3s1() {
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, pad_left_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, input_tensor_c_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::DepthwiseConv2d3x3() {
  use_lws_ = false;
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, stride_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, offset_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, input_c_block_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(16, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::DepthwiseConv2d() {
  use_lws_ = false;
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, stride_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, offset_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, offset_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, dilation_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(16, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(17, filter_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(18, filter_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(19, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Conv2dCommon() {
  use_lws_ = false;
  status_ = kernel_.setArg(0, c_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(1, w_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(2, nh_blk_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(3, *input_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(4, *filter_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(5, *bias_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(6, *output_image_p_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(7, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(8, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(9, input_c_block_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, output_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, filter_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, filter_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, stride_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(15, stride_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(16, pad_left_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(17, pad_up_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(18, dilation_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(19, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(20, *alpha_image_p_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Run() {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  // setArg
  (this->*impl_)();

  auto& context = ctx_->As<OpenCLContext>();
  /*
  status_ = context.cl_context()->RunKernel(
      kernel_, global_work_size_, local_work_size_, &event_);
  */

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 local_work_size_,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::PrintConvInfo() {
  const bool is_element_wise_bias =
      has_bias_ && conv_param_->output->dims() == conv_param_->bias->dims();

  VLOG(4) << "input_image_shape: " << input_image_w_ << "," << input_image_h_;
  VLOG(4) << "input_dims: " << conv_param_->x->dims();
  VLOG(4) << "filter_dims: " << conv_param_->filter->dims();
  if (has_bias_) {
    VLOG(4) << "bias_dims: " << conv_param_->bias->dims();
  }
  VLOG(4) << "output_dims: " << conv_param_->output->dims();
  VLOG(4) << "out_image_shape: " << output_image_w_ << ", " << output_image_h_;
  VLOG(4) << "paddings: " << pad_left_ << "," << pad_up_;
  VLOG(4) << "has bias: " << has_bias_;
  VLOG(4) << "is_element_wise_bias : " << is_element_wise_bias;
  VLOG(4) << "strides: " << stride_h_ << "," << stride_w_;
  VLOG(4) << "offset: ";
  VLOG(4) << "dilations.size : " << conv_param_->dilations->size();
  VLOG(4) << "dilations: " << dilation_h_ << ", " << dilation_w_;
  for (auto i = 0; i < global_work_size_.dimensions(); i++) {
    VLOG(4) << "global_work_size[" << i << "]: " << global_work_size_[i];
  }
  VLOG(4) << "groups_:" << groups_;

  LOG(INFO) << "================================";
  LOG(INFO) << "c_blk_=" << c_blk_ << ", w_blk_=" << w_blk_
            << ",nh_blk_=" << nh_blk_;
  //  LOG(INFO) << "input_image_p_:" << input_image_p_;
  //  LOG(INFO) << "filter_image_p_:" << filter_image_p_;
  //  LOG(INFO) << "bias_image_p_:" << bias_image_p_;
  //  LOG(INFO) << "output_image_p_:" << output_image_p_;

  LOG(INFO) << "stride_h_:" << stride_h_;
  LOG(INFO) << "stride_w_:" << stride_w_;

  LOG(INFO) << "dilation_h_:" << dilation_h_;
  LOG(INFO) << "dilation_w_:" << dilation_w_;

  LOG(INFO) << "pad_up_:" << pad_up_;
  LOG(INFO) << "pad_down_:" << pad_down_;
  LOG(INFO) << "pad_left_:" << pad_left_;
  LOG(INFO) << "pad_right_:" << pad_right_;

  LOG(INFO) << "offset_:" << offset_;
  LOG(INFO) << "groups_:" << groups_;
  LOG(INFO) << "relu_fused_:" << relu_fused_;
  LOG(INFO) << "has_bias_:" << has_bias_;

  LOG(INFO) << "input_tensor_n_:" << input_tensor_n_;
  LOG(INFO) << "input_tensor_c_:" << input_tensor_c_;
  LOG(INFO) << "input_tensor_h_:" << input_tensor_h_;
  LOG(INFO) << "input_tensor_w_:" << input_tensor_w_;
  LOG(INFO) << "input_image_h_:" << input_image_h_;
  LOG(INFO) << "input_image_w_:" << input_image_w_;
  LOG(INFO) << "input_c_block_:" << input_c_block_;

  LOG(INFO) << "output_tensor_n_:" << output_tensor_n_;
  LOG(INFO) << "output_tensor_c_:" << output_tensor_c_;
  LOG(INFO) << "output_tensor_h_:" << output_tensor_h_;
  LOG(INFO) << "output_tensor_w_:" << output_tensor_w_;
  LOG(INFO) << "output_image_h_:" << output_image_h_;
  LOG(INFO) << "output_image_w_:" << output_image_w_;

  LOG(INFO) << "filter_tensor_n_:" << filter_tensor_n_;
  LOG(INFO) << "filter_tensor_c_:" << filter_tensor_c_;
  LOG(INFO) << "filter_tensor_h_:" << filter_tensor_h_;
  LOG(INFO) << "filter_tensor_w_:" << filter_tensor_w_;
  LOG(INFO) << "filter_image_h_:" << filter_image_h_;
  LOG(INFO) << "filter_image_w_:" << filter_image_w_;

  LOG(INFO) << "bias_image_h_" << bias_image_h_;
  LOG(INFO) << "bias_image_w_" << bias_image_w_;
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// kARM(Mobile)
REGISTER_LITE_KERNEL(conv2d,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ConvImageCompute,
                     image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Prelu_alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ConvImageCompute,
                     image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Prelu_alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();

// kHost(PC)
REGISTER_LITE_KERNEL(conv2d,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ConvImageCompute,
                     image2d_pc)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Prelu_alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ConvImageCompute,
                     image2d_pc)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Prelu_alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();
#define LITE_WITH_LOG
