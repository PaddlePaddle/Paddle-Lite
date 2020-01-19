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

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class DepthwiseConv2dCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ConvParam;

  std::string doc() const override {
    return "DepthwiseConv2d using cl::Buffer, kFloat";
  }

  void PrepareForRun() override {
    const auto& param = *param_.get_mutable<param_t>();
    if (param.fuse_relu) {
      build_options_ += " -DRELU";
    }
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/depthwise_conv2d_kernel.cl", build_options_);
  }

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    auto x_dims = param.x->dims();
    auto filter_dims = param.filter->dims();
    auto output_dims = param.output->dims();
    auto paddings = *param.paddings;
    auto strides = param.strides;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* input_buf = param.x->data<float, cl::Buffer>();
    auto* filter_buf = param.filter->data<float, cl::Buffer>();
    auto* bias_buf = param.bias == nullptr
                         ? static_cast<cl::Buffer*>(nullptr)
                         : param.bias->data<float, cl::Buffer>();
    auto* output_buf =
        param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    cl_int status;
    auto numel = output_dims.production();
    int arg_idx = 0;
    status = kernel.setArg(arg_idx, static_cast<const int>(numel));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *input_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(output_dims[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(output_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(output_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(filter_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(filter_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(strides[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(strides[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(paddings[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(paddings[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *output_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *filter_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *bias_buf);
    CL_CHECK_FATAL(status);
    auto global_work_size = cl::NDRange(static_cast<size_t>(numel));
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    context.cl_wait_list()->emplace(output_buf, event_);
  }

 private:
  std::string kernel_func_name_{"depthwise_conv2d_3x3"};
  std::string build_options_{"-DCL_DTYPE=float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class DepthwiseConv2dComputeFP16Image
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ConvParam;

  std::string doc() const override {
    return "DepthwiseConv2d using cl::Image2D/kImageDefault, kFP16";
  }

  void PrepareForRun() override {
    const auto& param = *param_.get_mutable<param_t>();
    if (param.fuse_relu) {
      build_options_ += " -DRELU";
    }
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/depthwise_conv2d_kernel.cl", build_options_);
  }

  void Run() override {
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
    auto* input_img = param.x->data<int16_t, cl::Image2D>();
    auto* filter_img = param.filter->data<int16_t, cl::Image2D>();

    auto* bias_img = param.bias == nullptr
                         ? static_cast<cl::Image2D*>(nullptr)
                         : param.bias->data<int16_t, cl::Image2D>();

    auto image_shape = InitImageDimInfoWith(output_dims);

    auto* output_img = param.output->mutable_data<int16_t, cl::Image2D>(
        image_shape["width"], image_shape["height"]);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int c_block = (output_dims[1] + 3) / 4;
    int w = output_dims[3];
    int nh = output_dims[0] * output_dims[2];
    auto global_work_size = cl::NDRange(c_block, w, nh);

    LOG(INFO) << "setArg";
    LOG(INFO) << "c_block = " << c_block;
    LOG(INFO) << "w = " << w;
    LOG(INFO) << "nh = " << nh;

    LOG(INFO) << "strides = " << strides[0];
    LOG(INFO) << "offset = " << offset;
    LOG(INFO) << "dilations = " << dilations[0];
    LOG(INFO) << "input_c_block = " << input_c_block;
    LOG(INFO) << "x_dims[3] = " << x_dims[3];
    LOG(INFO) << "x_dims[2] = " << x_dims[2];
    LOG(INFO) << "output_dims[3] = " << output_dims[3];
    LOG(INFO) << "output_dims[2] = " << output_dims[2];

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

 private:
  std::string kernel_func_name_{"depth_conv2d_3x3"};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class DepthwiseConv2d3x3s1ComputeFP16Image
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ConvParam;

  std::string doc() const override {
    return "DepthwiseConv2d3x3s1 using cl::Image2D/kImageDefault, kFP16";
  }

  void PrepareForRun() override {
    const auto& param = *param_.get_mutable<param_t>();
    if (param.fuse_relu) {
      build_options_ += " -DRELU";
    }
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/depthwise_conv2d_kernel.cl", build_options_);
  }

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    auto x_dims = param.x->dims();
    auto filter_dims = param.filter->dims();
    auto output_dims = param.output->dims();
    auto paddings = *param.paddings;
    auto strides = param.strides;
    auto dilations = *param.dilations;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* input_img = param.x->data<int16_t, cl::Image2D>();
    auto* filter_img = param.filter->data<int16_t, cl::Image2D>();

    auto* bias_img = param.bias == nullptr
                         ? static_cast<cl::Image2D*>(nullptr)
                         : param.bias->data<int16_t, cl::Image2D>();

    auto image_shape = InitImageDimInfoWith(output_dims);

    auto* output_img = param.output->mutable_data<int16_t, cl::Image2D>(
        image_shape["width"], image_shape["height"]);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
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

 private:
  std::string kernel_func_name_{"depth_conv2d_3x3s1"};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class DepthwiseConv2dBasicComputeFP32Image
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFloat),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ConvParam;

  std::string doc() const override {
    return "DepthwiseConv2d basic using cl::Image2D/kImageDefault, kFloat32";
  }

  void PrepareForRun() override {
    const auto& param = *param_.get_mutable<param_t>();
    const bool has_bias = param.bias != nullptr;
    const bool is_element_wise_bias =
        has_bias && param.output->dims() == param.bias->dims();
    if (param.fuse_relu) {
      build_options_ += " -DRELU";
    }
    if (has_bias) {
      build_options_ += is_element_wise_bias ? " -DBIASE_ELE" : " -DBIASE_CH";
    }
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/depthwise_conv2d_basic_kernel.cl",
                                    build_options_);
  }

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    auto input_dims = param.x->dims();
    auto paddings = *param.paddings;
    auto strides = param.strides;
    auto* input_image = param.x->data<float, cl::Image2D>();
    auto* filter_image = param.filter->data<float, cl::Image2D>();
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

    LOG(INFO) << "============ depthwise conv2d params ============";
    LOG(INFO) << "input_image_shape: " << input_image_shape["width"] << ","
              << input_image_shape["height"];
    LOG(INFO) << "input_c_block: " << input_c_block;
    LOG(INFO) << "input_c: " << input_c;
    LOG(INFO) << "input_image: " << input_image;
    LOG(INFO) << "filter_dims: " << filter_dims;
    LOG(INFO) << "filter_image: " << filter_image;
    LOG(INFO) << "output_dims: " << output_dims;
    LOG(INFO) << "out_image_shape: " << out_image_shape["width"] << ", "
              << out_image_shape["height"];
    LOG(INFO) << "paddings: " << paddings[0] << "," << paddings[1];
    LOG(INFO) << "has bias: " << has_bias;
    LOG(INFO) << "is_element_wise_bias : " << is_element_wise_bias;
    LOG(INFO) << "strides: " << strides[0] << "," << strides[1];
    LOG(INFO) << "offset: " << offset;
    LOG(INFO) << "dilations.size : " << dilations.size();
    LOG(INFO) << "dilations: " << dilations[0] << ", " << dilations[1];
    LOG(INFO) << "default work size{c_block, w, nh}: "
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
      bias_image = param.bias->data<float, cl::Image2D>();
    }

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());
    LOG(INFO) << "kernel_key: " << kernel_key.str();
    LOG(INFO) << "kernel ready ... " << kernel_key.str();
    LOG(INFO) << "w: " << w;

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
      LOG(INFO) << "set bias_image: ";
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

    LOG(INFO) << "out_image: " << out_image;
    LOG(INFO) << "global_work_size[3D]: {" << global_work_size[0] << ","
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

 private:
  std::string kernel_func_name_{"depth_conv2d"};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};
}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(depthwise_conv2d,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::DepthwiseConv2dCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::DepthwiseConv2dComputeFP16Image,
    image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageNW))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
REGISTER_LITE_KERNEL(
    depthwise_conv2d_basic,
    kOpenCL,
    kFloat,
    kImageDefault,
    paddle::lite::kernels::opencl::DepthwiseConv2dBasicComputeFP32Image,
    image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageNW))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
