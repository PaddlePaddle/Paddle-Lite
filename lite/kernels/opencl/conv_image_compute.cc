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

#include <iomanip>
#include <sstream>
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"

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

  use_tune_ = CLRuntime::Global()->auto_tune();
  if (!is_mali) {
    use_tune_ = false;
  }
#ifdef LITE_WITH_LOG
  LOG(INFO) << "use_tune_" << use_tune_;
#endif

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

  bool pad_equal = ((pad_left_ == pad_up_) && (pad_up_ == pad_left_) &&
                    (pad_left_ == pad_right_));
  bool stride_equal = stride_h_ == stride_w_;
  bool dilation_equal = dilation_h_ == dilation_w_;

#ifdef LITE_WITH_LOG
  VLOG(3) << "Is arm mali  / " << (is_mali ? "Yes" : "No");
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

  CHECK(pad_equal && stride_equal && dilation_equal);
  CHECK_GE(conv_param_->dilations->size(), 2);
  CHECK(dilation_h_ == dilation_w_);
  CHECK_GE(conv_param_->paddings->size(), 2);
  CHECK(pad_left_ == pad_up_);
  CHECK_GE(conv_param_->strides.size(), 2);
  CHECK(stride_h_ == stride_w_);

  /*********************************************
   * Upload filter, bias to opencl device
   *********************************************/
  float* filter_cpu = conv_param_->filter->mutable_data<float>();
  filter_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
  tensor_hold_filter_image_ = std::unique_ptr<Tensor>(new Tensor);
  tensor_hold_bias_image_ = std::unique_ptr<Tensor>(new Tensor);

  if (filter_tensor_h_ == 1 && filter_tensor_h_ == 1) {
    if (input_tensor_c_ % 4 == 0) {
      kernel_func_names_.push_back("conv2d_1x1_simple");
    } else {
      kernel_func_names_.push_back("conv2d_1x1_opt");
    }
    kernel_func_paths_.push_back("image/conv2d_1x1_opt_kernel.cl");

    CLImageConverterNWBlock converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_h_ = filter_image_dims[1];
    filter_image_w_ = filter_image_dims[0];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});
    half_t* filter_image_data =
        tensor_hold_filter_image_->mutable_data<half_t>();

    converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
    filter_gpu_image_->mutable_data<half_t, cl::Image2D>(
        filter_image_w_, filter_image_h_, filter_image_data);

    impl_ = &ConvImageCompute::Conv2d1x1opt;
#define DEPTH_CONV_USE_SPL
#ifdef DEPTH_CONV_USE_SPL
  } else if (filter_tensor_c_ == 1 && input_tensor_c_ == output_tensor_c_ &&
             filter_tensor_h_ == 3 && filter_tensor_w_ == 3 && groups_ > 1) {
    // depth_conv2d_3x3s1, depth_conv2d_3x3
    if (stride_h_ == 1 && dilation_h_ == 1) {
      kernel_func_names_.push_back("depth_conv2d_3x3s1");
      impl_ = &ConvImageCompute::DepthwiseConv2d3x3s1;
    } else {
      kernel_func_names_.push_back("depth_conv2d_3x3");
      impl_ = &ConvImageCompute::DepthwiseConv2d3x3;
    }
    kernel_func_paths_.push_back("image/depthwise_conv2d_kernel.cl");

    CLImageConverterNWBlock converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_h_ = filter_image_dims[1];
    filter_image_w_ = filter_image_dims[0];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});

    half_t* filter_image_data =
        tensor_hold_filter_image_->mutable_data<half_t>();

    converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
    filter_gpu_image_->mutable_data<half_t, cl::Image2D>(
        filter_image_w_, filter_image_h_, filter_image_data);

#endif
  } else if (filter_tensor_c_ == 1 && input_tensor_c_ == output_tensor_c_
#ifdef DEPTH_CONV_USE_SPL
             &&
             filter_tensor_h_ != 3
#endif
#undef DEPTH_CONV_USE_SPL
             ) {
    // depth_conv2d
    kernel_func_names_.push_back("depth_conv2d");
    kernel_func_paths_.push_back("image/depthwise_conv2d_basic_kernel.cl");

    CLImageConverterNWBlock converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_h_ = filter_image_dims[1];
    filter_image_w_ = filter_image_dims[0];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});

    half_t* filter_image_data =
        tensor_hold_filter_image_->mutable_data<half_t>();

    converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
    filter_gpu_image_->mutable_data<half_t, cl::Image2D>(
        filter_image_w_, filter_image_h_, filter_image_data);

    impl_ = &ConvImageCompute::DepthwiseConv2d;
  } else if (filter_tensor_h_ == 3 && filter_tensor_w_ == 3) {
// #define CONV3x3OPT_FALL_BACK
#ifndef CONV3x3OPT_FALL_BACK
    // conv2d_3x3
    kernel_func_names_.push_back(input_tensor_n_ > 1 ? "conv2d_3x3_multi_batch"
                                                     : "conv2d_3x3_opt");
    kernel_func_paths_.push_back("image/conv2d_3x3_opt_kernel.cl");

    CLImageConverterFolder converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_h_ = filter_image_dims[1];
    filter_image_w_ = filter_image_dims[0];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});

    half_t* filter_image_data =
        tensor_hold_filter_image_->mutable_data<half_t>();

    converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
    filter_gpu_image_->mutable_data<half_t, cl::Image2D>(
        filter_image_w_, filter_image_h_, filter_image_data);

    impl_ = &ConvImageCompute::Conv2d3x3opt;
#else
    kernel_func_names_.push_back("conv2d_3x3");
    kernel_func_paths_.push_back("image/conv2d_3x3_kernel.cl");

    CLImageConverterFolder converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_h_ = filter_image_dims[1];
    filter_image_w_ = filter_image_dims[0];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});

    half_t* filter_image_data =
        tensor_hold_filter_image_->mutable_data<half_t>();

    converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
    filter_gpu_image_->mutable_data<half_t, cl::Image2D>(
        filter_image_w_, filter_image_h_, filter_image_data);

    impl_ = &ConvImageCompute::Conv2d3x3;
#endif
#undef CONV3x3OPT_FALL_BACK
  } else if (filter_tensor_h_ == 5 && filter_tensor_w_ == 5) {
#define CONV_5x5_OPT
#ifndef CONV_5x5_OPT
    // conv2d_5x5
    kernel_func_names_.push_back("conv2d_5x5");
    kernel_func_paths_.push_back("image/conv2d_5x5_kernel.cl");

    CLImageConverterFolder converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_h_ = filter_image_dims[1];
    filter_image_w_ = filter_image_dims[0];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});

    half_t* filter_image_data =
        tensor_hold_filter_image_->mutable_data<half_t>();

    converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
    filter_gpu_image_->mutable_data<half_t, cl::Image2D>(
        filter_image_w_, filter_image_h_, filter_image_data);

    impl_ = &ConvImageCompute::Conv2d5x5;
#else
    // conv2d_5x5_opt

    kernel_func_names_.push_back(input_tensor_n_ > 1 ? "conv2d_5x5_multi_batch"
                                                     : "conv2d_5x5_opt");
    kernel_func_paths_.push_back("image/conv2d_5x5_opt_kernel.cl");

    CLImageConverterFolder converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_h_ = filter_image_dims[1];
    filter_image_w_ = filter_image_dims[0];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});

    half_t* filter_image_data =
        tensor_hold_filter_image_->mutable_data<half_t>();

    converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
    filter_gpu_image_->mutable_data<half_t, cl::Image2D>(
        filter_image_w_, filter_image_h_, filter_image_data);

    impl_ = &ConvImageCompute::Conv2d5x5opt;
#endif
#undef CONV_5x5_OPT
  } else if (filter_tensor_h_ == 7 && filter_tensor_w_ == 7) {
#define CONV_7x7_OPT
#ifndef CONV_7x7_OPT
    // conv2d_7x7
    kernel_func_names_.push_back("conv2d_7x7");
    kernel_func_paths_.push_back("image/conv2d_7x7_kernel.cl");

    CLImageConverterFolder converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_h_ = filter_image_dims[1];
    filter_image_w_ = filter_image_dims[0];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});

    half_t* filter_image_data =
        tensor_hold_filter_image_->mutable_data<half_t>();

    converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
    filter_gpu_image_->mutable_data<half_t, cl::Image2D>(
        filter_image_w_, filter_image_h_, filter_image_data);

    impl_ = &ConvImageCompute::Conv2d7x7;

#else
    // conv2d_7x7
    kernel_func_names_.push_back(input_tensor_n_ > 1 ? "conv2d_7x7_multi_batch"
                                                     : "conv2d_7x7_opt");
    kernel_func_paths_.push_back("image/conv2d_7x7_opt_kernel.cl");

    CLImageConverterFolder converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_h_ = filter_image_dims[1];
    filter_image_w_ = filter_image_dims[0];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});

    half_t* filter_image_data =
        tensor_hold_filter_image_->mutable_data<half_t>();

    converter.NCHWToImage(filter_cpu, filter_image_data, filter_dims);
    filter_gpu_image_->mutable_data<half_t, cl::Image2D>(
        filter_image_w_, filter_image_h_, filter_image_data);

    impl_ = &ConvImageCompute::Conv2d7x7opt;
#endif
#undef CONV_7x7_OPT
  } else {
    LOG(FATAL) << "conv image compute not support this condition yet! ";
  }
  VLOG(1) << "kernel_func_names_[0]:" << kernel_func_names_[0]
          << " kernel_func_paths_[0]:" << kernel_func_paths_[0];

  // build options
  std::string build_options_single(" -DCL_DTYPE_half");
  // relu options
  VLOG(3) << "relu_fused_:" << relu_fused_
          << " conv_param_->activation_param.active_type:"
          << static_cast<int>(conv_param_->activation_param.active_type)
          << " conv_param_->activation_param.has_active:"
          << conv_param_->activation_param.has_active;
  if (conv_param_->activation_param.has_active) {
    if (conv_param_->activation_param.active_type ==
        lite_api::ActivationType::kRelu) {  // Note: judge using `relu_fused_`
                                            // also is ok
      build_options_single += " -DRELU";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kRelu6) {
      build_options_single += " -DRELU6";
    } else {
      LOG(FATAL) << "Unsupported activation type:"
                 << static_cast<int>(conv_param_->activation_param.active_type);
    }
  }
  GetGlobalWorkSize();

  // bias options
  const bool is_element_wise_bias =
      has_bias_ && conv_param_->output->dims() == conv_param_->bias->dims();
  if (has_bias_) {
    bias_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    build_options_single +=
        is_element_wise_bias ? " -DBIASE_ELE" : " -DBIASE_CH";

    // convert cpu buffer bias --> gpu image
    CLImageConverterFolder bias_converter;
    const DDim& bias_image_dims =
        bias_converter.InitImageDimInfoWith(conv_param_->bias->dims());
    bias_image_h_ = bias_image_dims[1];
    bias_image_w_ = bias_image_dims[0];
    tensor_hold_bias_image_->Resize(
        {1, bias_image_dims[0], bias_image_dims[1], 4});

    half_t* bias_image_data = tensor_hold_bias_image_->mutable_data<half_t>();

    float* bias_cpu_data = conv_param_->bias->mutable_data<float>();
    bias_converter.NCHWToImage(
        bias_cpu_data, bias_image_data, conv_param_->bias->dims());
    this->bias_gpu_image_->mutable_data<half_t, cl::Image2D>(
        bias_image_dims[0], bias_image_dims[1], bias_image_data);
    // convert cpu buffer bias --> gpu image --- end ----
  } else {
    bias_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    CLImageConverterFolder bias_converter;
    tensor_hold_bias_image_->Resize({1, 1, 1, 4});
    half_t* bias_image_data = tensor_hold_bias_image_->mutable_data<half_t>();
    this->bias_gpu_image_->mutable_data<half_t, cl::Image2D>(
        1, 1, bias_image_data);
  }

  // define image pointer for filter, bias
  input_image_p_ = conv_param_->x->data<half_t, cl::Image2D>();
  filter_image_p_ = filter_gpu_image_->data<half_t, cl::Image2D>();
  bias_image_p_ = bias_gpu_image_->data<half_t, cl::Image2D>();
  output_image_p_ = conv_param_->output->mutable_data<half_t, cl::Image2D>(
      output_image_w_, output_image_h_);

  build_options_.push_back(build_options_single);

  for (size_t i = 0; i < kernel_func_names_.size(); i++) {
    context.cl_context()->AddKernel(kernel_func_names_[i],
                                    kernel_func_paths_[i],
                                    build_options_[i],
                                    time_stamp_);
  }

  VLOG(4) << "global_work_size_[3D]: {" << global_work_size_[0] << ","
          << global_work_size_[1] << "," << global_work_size_[2] << "}";

  std::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0] << time_stamp_;
  kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  VLOG(4) << "kernel_key: " << kernel_key.str();
  VLOG(4) << "kernel ready ... " << kernel_key.str();
  size_t max_work_group_size = 0;
  kernel_.getWorkGroupInfo<size_t>(CLRuntime::Global()->device(),
                                   CL_KERNEL_WORK_GROUP_SIZE,
                                   &max_work_group_size);

  VLOG(4) << "max_work_group_size: " << max_work_group_size;

  if (max_work_group_size > 0 && use_lws_) {
    double min_tune_time = DBL_MAX;
    cl::NDRange best_local_work_size = context.cl_context()->LocalWorkSize(
        global_work_size_, max_work_group_size);
    VLOG(3) << "origin  :local_work_size_ : " << best_local_work_size[0] << " "
            << best_local_work_size[1] << " " << best_local_work_size[2];
    cl::NDRange last_local_work_size = cl::NDRange{
        static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>(0)};
    if (use_tune_) {
      for (size_t i = 1; i < 15; i++) {
        if (filter_tensor_h_ == 1 && filter_tensor_w_ == 1) {
          // todo use diff logics
          local_work_size_ = context.cl_context()->LocalWorkSizeTune(
              global_work_size_, max_work_group_size, i);
        } else {
          local_work_size_ = context.cl_context()->LocalWorkSizeTune(
              global_work_size_, max_work_group_size, i);
        }
        if (last_local_work_size[0] == local_work_size_[0] &&
            last_local_work_size[1] == local_work_size_[1] &&
            last_local_work_size[2] == local_work_size_[2]) {
          // skiped tuneed lws
          continue;
        }
        auto tune_time = this->Tune(10);
        if (min_tune_time > tune_time) {
          min_tune_time = tune_time;
          best_local_work_size = local_work_size_;
        }
        last_local_work_size = local_work_size_;
      }
      // reverse
      for (size_t i = 1; i < 15; i++) {
        if (filter_tensor_h_ == 1 && filter_tensor_w_ == 1) {
          // todo use diff logics
          local_work_size_ = context.cl_context()->LocalWorkSizeTuneReverse(
              global_work_size_, max_work_group_size, i);
        } else {
          local_work_size_ = context.cl_context()->LocalWorkSizeTuneReverse(
              global_work_size_, max_work_group_size, i);
        }
        if (last_local_work_size[0] == local_work_size_[0] &&
            last_local_work_size[1] == local_work_size_[1] &&
            last_local_work_size[2] == local_work_size_[2]) {
          // skiped tuneed lws
          continue;
        }
        auto tune_time = this->Tune(10);
        if (min_tune_time > tune_time) {
          min_tune_time = tune_time;
          best_local_work_size = local_work_size_;
        }
        last_local_work_size = local_work_size_;
      }
    }
    local_work_size_ = best_local_work_size;
    VLOG(3) << "chossen :local_work_size_ : " << local_work_size_[0] << " "
            << local_work_size_[1] << " " << local_work_size_[2];
    VLOG(4) << "local_work_size_[3D]: {" << local_work_size_[0] << ","
            << local_work_size_[1] << "," << local_work_size_[2] << "}";
  }
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
    if (kernel_func_names_.size() > 0 &&
        kernel_func_names_[0] == "conv2d_3x3") {
      groups_ = conv_param_->groups;
      if (filter_tensor_n_ == output_tensor_c_ &&
          filter_tensor_c_ == input_tensor_c_) {
        groups_ = 1;
      } else if (!(filter_tensor_n_ == input_tensor_c_ &&
                   filter_tensor_c_ == 1)) {
        groups_ = input_tensor_c_ / filter_tensor_c_;
      }
    }

    // define image pointer for input, output
    input_image_p_ = conv_param_->x->data<half_t, cl::Image2D>();
    output_image_p_ = conv_param_->output->mutable_data<half_t, cl::Image2D>(
        output_image_w_, output_image_h_);

    GetGlobalWorkSize();
  }
}

void ConvImageCompute::GetGlobalWorkSize() {
  if (kernel_func_names_.size() <= 0) return;
  // general input_c_block
  input_c_block_ = static_cast<int>(input_image_w_ / input_tensor_w_);

  // general gws
  auto output_dims = conv_param_->output->dims();
  const std::vector<size_t>& default_work_size =
      DefaultWorkSize(output_dims,
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

  if (kernel_func_names_[0] == "conv2d_1x1_simple" ||
      kernel_func_names_[0] == "conv2d_1x1_opt") {
    w_blk_ = maptofactor(default_w_blk_, 4);
    c_blk_ = default_c_blk_;
    nh_blk_ = default_nh_blk_;
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
  }
}

void ConvImageCompute::Conv2d1x1opt(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 local_work_size_,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);
  if (enable_tune) {
    CLRuntime::Global()->command_queue().finish();
  }
}

void ConvImageCompute::Conv2d3x3(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 cl::NullRange,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);
}

void ConvImageCompute::Conv2d3x3opt(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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

#ifdef LITE_WITH_LOG
  //  VLOG(4) << "out_image: " << out_image;
  VLOG(4) << "global_work_size_[3D]: {" << global_work_size_[0] << ","
          << global_work_size_[1] << "," << global_work_size_[2] << "}";
#endif

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 local_work_size_,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);
  if (enable_tune) {
    CLRuntime::Global()->command_queue().finish();
  }
}

void ConvImageCompute::Conv2d5x5(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 cl::NullRange,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);
  if (enable_tune) {
    CLRuntime::Global()->command_queue().finish();
  }
}

void ConvImageCompute::Conv2d5x5opt(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 local_work_size_,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);
  if (enable_tune) {
    CLRuntime::Global()->command_queue().finish();
  }
}

void ConvImageCompute::Conv2d7x7(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 cl::NullRange,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);
  if (enable_tune) {
    CLRuntime::Global()->command_queue().finish();
  }
}

void ConvImageCompute::Conv2d7x7opt(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 local_work_size_,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);

  if (enable_tune) {
    CLRuntime::Global()->command_queue().finish();
  }
}

void ConvImageCompute::DepthwiseConv2d3x3s1(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 local_work_size_,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);

  if (enable_tune) {
    CLRuntime::Global()->command_queue().finish();
  }
}

void ConvImageCompute::DepthwiseConv2d3x3(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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
  status_ = kernel_.setArg(9, dilation_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(10, input_c_block_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(11, input_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(12, input_tensor_h_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(13, output_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(14, output_tensor_h_);
  CL_CHECK_FATAL(status_);

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 cl::NullRange,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);

  if (enable_tune) {
    CLRuntime::Global()->command_queue().finish();
  }
}

void ConvImageCompute::DepthwiseConv2d(bool enable_tune) {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif
  auto& context = ctx_->As<OpenCLContext>();

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
  status_ = kernel_.setArg(15, filter_tensor_w_);
  CL_CHECK_FATAL(status_);
  status_ = kernel_.setArg(16, filter_tensor_h_);
  CL_CHECK_FATAL(status_);

  status_ = EnqueueNDRangeKernel(context,
                                 kernel_,
                                 cl::NullRange,
                                 global_work_size_,
                                 cl::NullRange,
                                 nullptr,
                                 event_);
  CL_CHECK_FATAL(status_);

  if (enable_tune) {
    CLRuntime::Global()->command_queue().finish();
  }
}

void ConvImageCompute::Run() { (this->*impl_)(false); }

void ConvImageCompute::PrintConvInfo() {
  const bool is_element_wise_bias =
      has_bias_ && conv_param_->output->dims() == conv_param_->bias->dims();

  VLOG(4) << "input_image_shape: " << input_image_w_ << "," << input_image_h_;
  //  VLOG(4) << "input_image: " << input_image_p_;
  VLOG(4) << "input_dims: " << conv_param_->x->dims();
  VLOG(4) << "filter_dims: " << conv_param_->filter->dims();
  //  VLOG(4) << "filter_image: " << filter_image;
  VLOG(4) << "output_dims: " << conv_param_->output->dims();
  VLOG(4) << "out_image_shape: " << output_image_w_ << ", " << output_image_h_;
  VLOG(4) << "paddings: " << pad_left_ << "," << pad_up_;
  VLOG(4) << "has bias: " << has_bias_;
  VLOG(4) << "is_element_wise_bias : " << is_element_wise_bias;
  VLOG(4) << "strides: " << stride_h_ << "," << stride_w_;
  VLOG(4) << "offset: ";
  VLOG(4) << "dilations.size : " << conv_param_->dilations->size();
  VLOG(4) << "dilations: " << dilation_h_ << ", " << dilation_w_;
  VLOG(4) << "global_work_size_[3D]: {" << global_work_size_[0] << ","
          << global_work_size_[1] << "," << global_work_size_[2] << "}";
}

double ConvImageCompute::Tune(int times) {
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start = GetCurrentUS();
  for (size_t i = 0; i < times; i++) {
    (this->*impl_)(true);
  }
  auto time_diff = (GetCurrentUS() - start) / times;
  return time_diff;
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

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
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
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
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
#define LITE_WITH_LOG
