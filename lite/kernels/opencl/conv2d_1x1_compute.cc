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

#define USE_BUFFER_FOR_CONV1x1_BIAS
class Conv2d1x1Image2DCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ConvParam;

  void PrepareForRun() override {
    LOG(INFO) << "PrepareForRun  ...";
    const auto& param = *param_.get_mutable<param_t>();
    if (param.fuse_relu) {
      build_options_ += " -DRELU";
    }

    const bool has_bias = param.bias != nullptr;
    const bool is_element_wise_bias =
        has_bias && param.output->dims() == param.bias->dims();
    if (has_bias) {
      build_options_ += is_element_wise_bias ? " -DBIASE_ELE" : " -DBIASE_CH";
    }
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/conv_1x1_kernel.cl", build_options_);
    LOG(INFO) << "PrepareForRun  Ready";
  }

  void Run() override {
    LOG(INFO) << "opencl conv 1x1 run begin  ...";
    LOG(INFO) << "param_.get_mutable<param_t>  ...";
    const auto& param = *param_.get_mutable<param_t>();

    LOG(INFO) << "get param dims  ...";
    auto input_dims = param.x->dims();
    CHECK_GE(input_dims.size(), 4);
    LOG(INFO) << "input_dims: " << input_dims;

    int input_width = input_dims[3];
    int input_height = input_dims[2];

    auto filter_dims = param.filter->dims();

    LOG(INFO) << "filter_dims: " << filter_dims;

    auto output_dims = param.output->dims();

    LOG(INFO) << "output_dims: " << output_dims;

    int output_width = output_dims[3];
    int output_height = output_dims[2];
    // mute output image
    auto out_image_shape = InitImageDimInfoWith(output_dims);

    LOG(INFO) << "out_image_shape: " << out_image_shape["width"] << ", "
              << out_image_shape["height"];

    auto* out_image = param.output->mutable_data<float, cl::Image2D>(
        out_image_shape["width"], out_image_shape["height"]);

    // gen default_work_size

    const std::vector<size_t>& default_work_size =
        DefaultWorkSize(output_dims,
                        DDim(std::vector<DDim::value_type>{
                            static_cast<int64_t>(out_image_shape["width"]),
                            static_cast<int64_t>(out_image_shape["height"])}));

    int c_block = default_work_size[0];
    int w = default_work_size[1];
    int nh = default_work_size[2];
    LOG(INFO) << "default work size: "
              << "{" << c_block << ", " << w << ", " << nh << ""
              << "}";

    auto paddings = *param.paddings;
    LOG(INFO) << "paddings: " << paddings[0] << "," << paddings[1];

    auto strides = param.strides;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* input_image = param.x->data<float, cl::Image2D>();
    auto* filter_image = param.filter->data<float, cl::Image2D>();

    // handle bias  use buffer for channel wise , use image for element wise
    const bool has_bias = param.bias != nullptr;
    const bool is_element_wise_bias =
        has_bias && param.output->dims() == param.bias->dims();

    LOG(INFO) << "has bias: " << has_bias;
    LOG(INFO) << "is_element_wise_bias : " << is_element_wise_bias;
    LOG(INFO) << "get kernel  ...";
    const cl::Buffer* bias_buf = nullptr;
    const cl::Image2D* bias_image = nullptr;
    if (has_bias) {
#ifndef USE_BUFFER_FOR_CONV1x1_BIAS
      is_element_wise_bias
          ? (bias_image = param.bias->data<float, cl::Image2D>())
          : (bias_buf = param.bias->data<float, cl::Buffer>());
#else
      bias_image = param.bias->data<float, cl::Image2D>();
#endif
    }
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;

    LOG(INFO) << "kernel_key: " << kernel_key.str();
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    LOG(INFO) << "kernel ready ... " << kernel_key.str();
    cl_int status;
    auto numel = output_dims.production();
    int arg_idx = 0;

    status = kernel.setArg(arg_idx, c_block);
    CL_CHECK_FATAL(status);

    int maped_w = maptofactor(w, 4);
    LOG(INFO) << "maped_w: " << maped_w;

    status = kernel.setArg(++arg_idx, maped_w);
    CL_CHECK_FATAL(status);

    status = kernel.setArg(++arg_idx, nh);
    LOG(INFO) << "nh: " << nh;

    CL_CHECK_FATAL(status);

    status = kernel.setArg(++arg_idx, *input_image);
    LOG(INFO) << "input_image: ";

    CL_CHECK_FATAL(status);

    status = kernel.setArg(++arg_idx, *filter_image);
    LOG(INFO) << "filter_image: ";

    CL_CHECK_FATAL(status);

    if (has_bias) {
#ifndef USE_BUFFER_FOR_CONV1x1_BIAS
      if (is_element_wise_bias != 0) {
        LOG(INFO) << "set bias_image: ";
        status = kernel.setArg(++arg_idx, *bias_image);
      } else {
        LOG(INFO) << "set bias_buf: ";
        status = kernel.setArg(++arg_idx, *bias_buf);
      }
#else
      status = kernel.setArg(++arg_idx, *bias_image);
#endif
      CL_CHECK_FATAL(status);
    }

    status = kernel.setArg(++arg_idx, *out_image);
    LOG(INFO) << "out_image: ";

    CL_CHECK_FATAL(status);

    CHECK_GE(strides.size(), 2);
    CHECK(strides[0] == strides[1]);
    status = kernel.setArg(++arg_idx, strides[0]);
    LOG(INFO) << "strides: " << strides[0] << "," << strides[1];

    CL_CHECK_FATAL(status);

    CHECK_GE(paddings.size(), 2);
    CHECK(paddings[0] == paddings[1]);

    int offset = static_cast<int>(param.filter->dims()[2]) / 2 -
                 static_cast<int>(paddings[0]);
    LOG(INFO) << "offset: " << offset;

    status = kernel.setArg(++arg_idx, offset);
    CL_CHECK_FATAL(status);

    // calc input_c_block
    auto input_image_shape = InitImageDimInfoWith(input_dims);
    LOG(INFO) << "input_image_shape: " << input_image_shape["width"] << ","
              << input_image_shape["height"];

    int input_c_block = input_image_shape["width"] / input_dims[3];
    LOG(INFO) << "input_c_block: " << input_c_block;

    status = kernel.setArg(++arg_idx, input_c_block);
    CL_CHECK_FATAL(status);

    int input_c = input_dims[1];
    LOG(INFO) << "input_c: " << input_c;

    status = kernel.setArg(++arg_idx, input_c);
    CL_CHECK_FATAL(status);

    auto dilations = *param.dilations;
    LOG(INFO) << "dilations.size : " << dilations.size();
    LOG(INFO) << "dilations: " << dilations[0] << ", " << dilations[1];

    CHECK_GE(dilations.size(), 2);
    CHECK(dilations[0] == dilations[1]);
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

    // clac gloabl_work_size
    auto global_work_size =
        cl::NDRange{static_cast<size_t>(default_work_size.data()[0]),
                    static_cast<size_t>(maped_w),
                    static_cast<size_t>(default_work_size.data()[2])};

    LOG(INFO) << "global_work_size  :" << global_work_size;

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
  std::string kernel_func_name_{"conv_1x1"};
  std::string build_options_{"-DCL_DTYPE=float "};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d_1x1,
                     kOpenCL,
                     kFloat,
                     kNHWC,
                     paddle::lite::kernels::opencl::Conv2d1x1Image2DCompute,
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
