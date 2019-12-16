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
  std::string kernel_func_name_{"depthwise_conv2d"};
  std::string build_options_{"-DCL_DTYPE=float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class DepthwiseConv2dComputeFP16Image
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ConvParam;

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
    auto* input_img = param.x->data<float, cl::Image2D>();
    auto* filter_img = param.filter->data<float, cl::Image2D>();

    auto* bias_img = param.bias == nullptr
                         ? static_cast<cl::Image2D*>(nullptr)
                         : param.bias->data<float, cl::Image2D>();

    auto image_shape = InitImageDimInfoWith(output_dims);

    auto* output_img = param.output->mutable_data<float, cl::Image2D>(
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
  std::string kernel_func_name_{"depth_conv_3x3"};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class DepthwiseConv2d3x3s1ComputeFP16Image
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ConvParam;

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
    auto* input_img = param.x->data<float, cl::Image2D>();
    auto* filter_img = param.filter->data<float, cl::Image2D>();

    auto* bias_img = param.bias == nullptr
                         ? static_cast<cl::Image2D*>(nullptr)
                         : param.bias->data<float, cl::Image2D>();

    auto image_shape = InitImageDimInfoWith(output_dims);

    auto* output_img = param.output->mutable_data<float, cl::Image2D>(
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
  std::string kernel_func_name_{"depth_conv_3x3s1"};
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
    kFloat,
    kNHWC,
    paddle::lite::kernels::opencl::DepthwiseConv2dComputeFP16Image,
    image2d)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Bias",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
