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
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class PoolCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::PoolParam;

  std::string doc() const override { return "Pool using cl::Buffer, kFloat"; }

  void PrepareForRun() override {
    const auto& param = *param_.get_mutable<param_t>();
    kernel_func_name_ += param.pooling_type;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/pool_kernel.cl", build_options_);
  }

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    const auto& in_dims = param.x->dims();
    const auto& out_dims = param.output->dims();
    const std::string pooling_type = param.pooling_type;
    const bool global_pooling = param.global_pooling;
    std::vector<int> paddings = *param.paddings;
    std::vector<int> strides = param.strides;
    std::vector<int> ksize = param.ksize;
    if (global_pooling) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[2 * i] = 0;
        paddings[2 * i + 1] = 0;
        ksize[i] = static_cast<int>(in_dims[i + 2]);
      }
    }
    bool pads_equal =
        (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]);
    if (!pads_equal) {
      LOG(FATAL)
          << "padding requires pad_left == pad_right, pad_top == pad_bottom";
    }
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* input_buf = param.x->data<float, cl::Buffer>();
    auto* output_buf =
        param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());
    cl_int status;
    auto numel = out_dims.production();
    int arg_idx = 0;
    status = kernel.setArg(arg_idx, static_cast<const int>(numel));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *input_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(ksize[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(ksize[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(strides[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(strides[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(paddings[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(paddings[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *output_buf);
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
  std::string kernel_func_name_{"pool_"};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class PoolComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                             PRECISION(kFloat),
                                             DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::PoolParam;

  std::string doc() const override { return "Pool using cl::Image2D, kFloat"; }

  void PrepareForRun() override {
    const auto& param = *param_.get_mutable<param_t>();
    kernel_func_name_ += param.pooling_type;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/pool_kernel.cl", build_options_);
  }

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    const auto& in_dims = param.x->dims();
    const auto& out_dims = param.output->dims();
    const std::string pooling_type = param.pooling_type;
    const bool global_pooling = param.global_pooling;
    std::vector<int> paddings = *param.paddings;
    std::vector<int> strides = param.strides;
    std::vector<int> ksize = param.ksize;
    if (global_pooling) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[2 * i] = 0;
        paddings[2 * i + 1] = 0;
        ksize[i] = static_cast<int>(in_dims[i + 2]);
      }
    }
    bool pads_equal =
        (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]);
    if (!pads_equal) {
      LOG(FATAL)
          << "padding requires pad_left == pad_right, pad_top == pad_bottom";
    }
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto* x_img = param.x->data<float, cl::Image2D>();
    LOG(INFO) << "x_image" << x_img;

    auto out_image_shape = InitImageDimInfoWith(out_dims);
    LOG(INFO) << "out_image_shape = " << out_image_shape["width"] << " "
              << out_image_shape["height"];
    auto* out_img = param.output->mutable_data<float, cl::Image2D>(
        out_image_shape["width"], out_image_shape["height"]);
    LOG(INFO) << "out_image" << out_img;

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int c_block = (out_dims[1] + 3) / 4;
    int w = out_dims[3];
    int nh = out_dims[0] * out_dims[2];
    auto global_work_size = cl::NDRange(c_block, w, nh);

    cl_int status;
    int arg_idx = 0;
    status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(ksize[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(ksize[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(strides[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(strides[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(paddings[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(paddings[0]));
    CL_CHECK_FATAL(status);

    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    context.cl_wait_list()->emplace(out_img, event_);
  }

 private:
  std::string kernel_func_name_{"pool_"};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// REGISTER_LITE_KERNEL(pool2d,
//                      kOpenCL,
//                      kFloat,
//                      kNCHW,
//                      paddle::lite::kernels::opencl::PoolCompute,
//                      def)
//     .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
//     .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
//     .Finalize();

REGISTER_LITE_KERNEL(pool2d,
                     kOpenCL,
                     kFloat,
                     kImageDefault,
                     paddle::lite::kernels::opencl::PoolComputeImage2D,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
