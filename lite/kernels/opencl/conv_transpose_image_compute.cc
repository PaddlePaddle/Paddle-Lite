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

#include "lite/kernels/opencl/conv_transpose_image_compute.h"
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void ConvTransposeImageCompute::PrepareForRun() {
  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  const bool is_mali = context.cl_context()->IsArmMali();

  conv_param_ = param_.get_mutable<param_t>();
  auto x_dims = conv_param_->x->dims();
  input_tensor_n_ = x_dims[0];
  input_tensor_c_ = x_dims[1];
  input_tensor_h_ = x_dims[2];
  input_tensor_w_ = x_dims[3];

  auto output_dims = conv_param_->output->dims();
  output_tensor_n_ = output_dims[0];
  output_tensor_c_ = output_dims[1];
  output_tensor_h_ = output_dims[2];
  output_tensor_w_ = output_dims[3];

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

  /*********************************************
   * Upload filter, bias to opencl device
   *********************************************/
  filter_tensor_h_ = dilation_h_ * (filter_tensor_h_ - 1) + 1;
  filter_tensor_w_ = dilation_w_ * (filter_tensor_w_ - 1) + 1;
  auto* filter_cpu = conv_param_->filter->mutable_data<float>();

  filter_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
  tensor_hold_filter_image_ = std::unique_ptr<Tensor>(new Tensor);
  tensor_hold_bias_image_ = std::unique_ptr<Tensor>(new Tensor);
  // build options
  std::string build_options_single{""};
  if (groups_ == 1) {
    std::string kernel_name = "conv2d_transpose";
    kernel_func_names_.push_back(kernel_name);

    CLImageConverterNBlock converter;
    const DDim& filter_image_dims = converter.InitImageDimInfoWith(filter_dims);
    filter_image_w_ = filter_image_dims[0];  // ((C + 3) / 4) * 4;
    filter_image_h_ = filter_image_dims[1];  // ((N + 3) / 4) * H * W;
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});
    auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);

    converter.NCHWToImage(
        reinterpret_cast<float*>(filter_cpu), filter_image_data, filter_dims);
    MUTABLE_DATA_GPU(
        filter_gpu_image_, filter_image_w_, filter_image_h_, filter_image_data);
  } else if ((groups_ == input_tensor_c_) && (groups_ == output_tensor_c_)) {
    // for depthwise conv transpose
    std::string kernel_name = "conv2d_transpose";
    build_options_single += " -DIS_DEPTHWISE ";
    kernel_func_names_.push_back(kernel_name);

    DDimLite filter_trans_dims{
        {filter_dims[1], filter_dims[0], filter_dims[2], filter_dims[3]}};
    CLImageConverterDefault converter;
    const DDim& filter_image_dims =
        converter.InitImageDimInfoWith(filter_trans_dims);
    filter_image_w_ = filter_image_dims[0];
    filter_image_h_ = filter_image_dims[1];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});
    auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);

    converter.NCHWToImage(reinterpret_cast<float*>(filter_cpu),
                          filter_image_data,
                          filter_trans_dims);
    MUTABLE_DATA_GPU(
        filter_gpu_image_, filter_image_w_, filter_image_h_, filter_image_data);
  } else if (groups_ > 1) {
    CHECK_EQ(filter_tensor_n_ % groups_, 0);
    std::string kernel_name = "group_conv2d_transpose";
    is_group_conv_ = true;
    kernel_func_names_.push_back(kernel_name);

    DDimLite filter_trans_dims{
        {filter_dims[0], filter_dims[1], filter_dims[2], filter_dims[3]}};
    CLImageConverterDefault converter;
    const DDim& filter_image_dims =
        converter.InitImageDimInfoWith(filter_trans_dims);
    filter_image_w_ = filter_image_dims[0];
    filter_image_h_ = filter_image_dims[1];
    tensor_hold_filter_image_->Resize({1, filter_image_w_, filter_image_h_, 4});
    auto* filter_image_data = MUTABLE_DATA_CPU(tensor_hold_filter_image_);
    converter.NCHWToImage(reinterpret_cast<float*>(filter_cpu),
                          filter_image_data,
                          filter_trans_dims);
    MUTABLE_DATA_GPU(
        filter_gpu_image_, filter_image_w_, filter_image_h_, filter_image_data);
  } else {
    LOG(FATAL)
        << "conv2d_transpose image compute not support this condition yet! "
        << groups_ << " " << input_tensor_c_ << " " << output_tensor_c_;
  }

  // bias options
  if (has_bias_) {
    bias_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    build_options_single += " -DBIASE_CH";

    // convert cpu buffer bias --> gpu image --- begin ---
    CLImageConverterFolder bias_converter;
    const DDim& bias_image_dims =
        bias_converter.InitImageDimInfoWith(conv_param_->bias->dims());
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
    // convert cpu buffer bias --> gpu image --- end ----
  } else {
    bias_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    CLImageConverterFolder bias_converter;
    tensor_hold_bias_image_->Resize({1, 1, 1, 4});
    auto* bias_image_data = DATA_GPU(tensor_hold_bias_image_);
    MUTABLE_DATA_GPU(bias_gpu_image_, 1, 1, bias_image_data);
  }

  // define image pointer for filter, bias
  filter_image_p_ = DATA_GPU(filter_gpu_image_);
  bias_image_p_ = DATA_GPU(bias_gpu_image_);

  // relu options
  VLOG(3) << "relu_fused_:" << relu_fused_
          << " conv_param_->activation_param.active_type:"
          << static_cast<int>(conv_param_->activation_param.active_type)
          << " conv_param_->activation_param.has_active:"
          << conv_param_->activation_param.has_active;
  if (conv_param_->activation_param.has_active) {
    if (conv_param_->activation_param.active_type ==
        lite_api::ActivationType::kRelu) {
      build_options_single += " -DRELU";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kSigmoid) {
      build_options_single += " -DSIGMOID";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kTanh) {
      build_options_single += " -DTANH";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kSwish) {
      std::string scale =
          std::to_string(conv_param_->activation_param.swish_scale);
      build_options_single += " -DSWISH -DACT_SCALE=" + scale + "f";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kAbs) {
      build_options_single += " -DABS";
    } else if (conv_param_->activation_param.active_type ==
               lite_api::ActivationType::kExp) {
      build_options_single += " -DEXP";
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
    } else {
      LOG(FATAL) << "Unsupported activation type:"
                 << static_cast<int>(conv_param_->activation_param.active_type);
    }
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

  kernel_func_paths_.push_back("image/conv2d_transpose_kernel.cl");
  VLOG(1) << "kernel_func_names_[0]:" << kernel_func_names_[0]
          << " kernel_func_paths_[0]:" << kernel_func_paths_[0];

  build_options_.push_back(build_options_single);
  for (size_t i = 0; i < kernel_func_names_.size(); i++) {
    context.cl_context()->AddKernel(kernel_func_names_[i],
                                    kernel_func_paths_[i],
                                    build_options_[i],
                                    time_stamp_);
  }

  std::stringstream kernel_key;
  kernel_key << kernel_func_names_[0] << build_options_[0] << time_stamp_;
  kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  VLOG(4) << "kernel_key: " << kernel_key.str();
}

void ConvTransposeImageCompute::ReInitWhenNeeded() {
  conv_param_ = param_.get_mutable<param_t>();
  auto x_dims = conv_param_->x->dims();
#ifdef LITE_WITH_LOG
  VLOG(4) << "is_first_epoch_for_run_:" << is_first_epoch_for_run_
          << ", last_input_dims_:" << last_input_dims_ << ", x_dims:" << x_dims;
#endif

  if (is_first_epoch_for_run_ || last_input_dims_ != x_dims) {
    is_first_epoch_for_run_ = false;
    last_input_dims_ = x_dims;

    auto x_image_shape = InitImageDimInfoWith(x_dims);
    input_image_h_ = x_image_shape["height"];
    input_image_w_ = x_image_shape["width"];

    auto output_dims = conv_param_->output->dims();
    auto output_image_shape = InitImageDimInfoWith(output_dims);
    output_image_h_ = output_image_shape["height"];
    output_image_w_ = output_image_shape["width"];

    CHECK_GE(x_dims.size(), 4);
    CHECK_GE(output_dims.size(), 4);

    // define image pointer for input, output
    input_image_p_ = DATA_GPU(conv_param_->x);
    output_image_p_ = MUTABLE_DATA_GPU(
        conv_param_->output, output_image_w_, output_image_h_, nullptr);

    SetGlobalWorkSize();
  }
}

void ConvTransposeImageCompute::SetGlobalWorkSize() {
  auto out_dims = conv_param_->output->dims();
  auto gws = DefaultGlobalWorkSize(out_dims,
                                   DDim(std::vector<DDim::value_type>{
                                       static_cast<int64_t>(output_image_w_),
                                       static_cast<int64_t>(output_image_h_)}));
  global_work_size_ = cl::NDRange{static_cast<size_t>(gws[0]),
                                  static_cast<size_t>(gws[1]),
                                  static_cast<size_t>(gws[2])};
  LOG(INFO) << "global_work_size_: " << gws[0] << " " << gws[1] << " "
            << gws[2];
}

void ConvTransposeImageCompute::SetArgs() {
  const int pad_w = filter_tensor_w_ - 1 - pad_left_;
  const int pad_h = filter_tensor_h_ - 1 - pad_up_;
  const int align_w = stride_w_ - 1 - pad_w;
  const int align_h = stride_h_ - 1 - pad_h;
  LOG(INFO) << "pad_w, pad_h: " << pad_w << " " << pad_h;
  LOG(INFO) << "align_w, align_h: " << align_w << " " << align_h;
  cl_int2 pad_wh = {pad_w, pad_h};
  cl_int2 align_wh = {align_w, align_h};

  cl_int2 input_wh = {input_tensor_w_, input_tensor_h_};
  cl_int2 output_wh = {output_tensor_w_, output_tensor_h_};
  cl_int2 filter_wh = {filter_tensor_w_, filter_tensor_h_};
  cl_int2 stride_wh = {stride_w_, stride_h_};
  cl_int2 dilation_wh = {dilation_w_, dilation_h_};
  cl_int2 filter_prev_wh = {
      static_cast<cl_int>(conv_param_->filter->dims()[3]),
      static_cast<cl_int>(conv_param_->filter->dims()[2])};
  auto kernel = &kernel_;

  uint32_t idx = 0;
  cl_int status;
  for (auto i = 0; i < global_work_size_.dimensions(); i++) {
    status = kernel->setArg(idx++, static_cast<int32_t>(global_work_size_[i]));
    CL_CHECK_FATAL(status);
  }
  kernel->setArg(idx++, *input_image_p_);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, *filter_image_p_);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, *bias_image_p_);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, *output_image_p_);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, input_wh);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, output_wh);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, stride_wh);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, align_wh);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, pad_wh);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, filter_wh);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, dilation_wh);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, filter_prev_wh);
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++,
                 static_cast<int32_t>(filter_tensor_w_ * filter_tensor_h_));
  CL_CHECK_FATAL(status);
  kernel->setArg(idx++, static_cast<int32_t>(maptofactor(input_tensor_c_, 4)));
  CL_CHECK_FATAL(status);
  if (is_group_conv_) {
    int in_channels_per_group = input_tensor_c_ / groups_;
    kernel->setArg(idx++, in_channels_per_group);
    CL_CHECK_FATAL(status);
    int out_channels_per_group = output_tensor_c_ / groups_;
    kernel->setArg(idx++, out_channels_per_group);
    CL_CHECK_FATAL(status);
    VLOG(4) << "in_per_group: " << in_channels_per_group
            << ", out_per_group: " << out_channels_per_group;
  }
}

void ConvTransposeImageCompute::Run() {
#ifdef LITE_WITH_LOG
  PrintConvInfo();
#endif

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);

  SetArgs();

  cl_int status = EnqueueNDRangeKernel(context,
                                       kernel_,
                                       cl::NullRange,
                                       global_work_size_,
                                       local_work_size_,
                                       nullptr,
                                       event_);
  CL_CHECK_FATAL(status);
}

#ifdef LITE_WITH_LOG
void ConvTransposeImageCompute::PrintConvInfo() {
  VLOG(4) << "input_dims: " << conv_param_->x->dims();
  VLOG(4) << "filter_dims: " << conv_param_->filter->dims();
  if (has_bias_) {
    VLOG(4) << "bias_dims: " << conv_param_->bias->dims();
  }
  VLOG(4) << "output_dims: " << conv_param_->output->dims();
  VLOG(4) << "input_image_shape: " << input_image_w_ << "," << input_image_h_;
  VLOG(4) << "filter_image_shape: " << filter_image_w_ << ","
          << filter_image_h_;
  VLOG(4) << "out_image_shape: " << output_image_w_ << ", " << output_image_h_;
  size_t w, h;
  output_image_p_->getImageInfo(CL_IMAGE_WIDTH, &w);
  output_image_p_->getImageInfo(CL_IMAGE_HEIGHT, &h);
  VLOG(4) << "out_image_shape getImageInfo: " << w << ", " << h;
  VLOG(4) << "global_work_size_[3D]: {" << global_work_size_[0] << ","
          << global_work_size_[1] << "," << global_work_size_[2] << "}";
  VLOG(4) << "================================";
  VLOG(4) << "has bias: " << has_bias_;
  VLOG(4) << "relu_fused_:" << relu_fused_;
  VLOG(4) << "strides: " << stride_h_ << "," << stride_w_;
  VLOG(4) << "dilations: " << dilation_h_ << ", " << dilation_w_;
  VLOG(4) << "groups_:" << groups_;
  VLOG(4) << "pad_up_:" << pad_up_;
  VLOG(4) << "pad_down_:" << pad_down_;
  VLOG(4) << "pad_left_:" << pad_left_;
  VLOG(4) << "pad_right_:" << pad_right_;
}
#endif

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d_transpose,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ConvTransposeImageCompute,
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
    .BindPaddleOpVersion("conv2d_transpose", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d_transpose,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::ConvTransposeImageCompute,
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
    .BindPaddleOpVersion("depthwise_conv2d_transpose", 1)
    .Finalize();
