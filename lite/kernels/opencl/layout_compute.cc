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

#include <memory>
#include <string>
#include "lite/api/paddle_place.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/type_system.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class LayoutComputeBufferChwToImage2DHwc
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/layout_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = Param<param_t>();
    auto* x_data = param.x->data<float, cl::Buffer>();
    auto x_dims = param.x->dims();
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto* y_data = param.y->mutable_data<float, cl::Image2D>(
        image_shape["width"], image_shape["height"]);
    auto y_dims = param.y->dims();

    // out info
    std::vector<size_t> new_dims = {1, 1, 1, 1};
    for (int tidx = 0; tidx < x_dims.size(); ++tidx) {
      new_dims[4 - x_dims.size() + tidx] = x_dims[tidx];
    }
    const int out_C = new_dims[1];
    const int out_H = new_dims[2];
    const int out_W = new_dims[3];
    const int Stride2 = out_C * out_H * out_W;
    const int Stride1 = out_H * out_W;
    const int Stride0 = out_W;

    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];
    VLOG(4) << "new_dims[" << new_dims.size() << "D]:" << new_dims[0] << " "
            << new_dims[1] << " " << new_dims[2] << " " << new_dims[3];
    VLOG(4) << "out_C:" << out_C;
    VLOG(4) << "out_H:" << out_H;
    VLOG(4) << "out_W:" << out_W;
    VLOG(4) << "Stride2:" << Stride2;
    VLOG(4) << "Stride1:" << Stride1;
    VLOG(4) << "Stride0:" << Stride0;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_H));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_W));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_C));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(Stride0));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(Stride1));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(Stride2));
    CL_CHECK_FATAL(status);

    VLOG(4) << "gws:[3D]" << ((new_dims[1] + 3) / 4) << " " << new_dims[3]
            << " " << (new_dims[0] * new_dims[2]);
    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>((new_dims[1] + 3) / 4),
                    static_cast<cl::size_type>(new_dims[3]),
                    static_cast<cl::size_type>(new_dims[0] * new_dims[2])};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    // TODO(ysh329): io_copy(device->host) jammed if emplace to `cl_wait_list`
    // context.cl_wait_list()->emplace(y_data, event_);
    context.cl_context()->GetCommandQueue().finish();
  }

  std::string doc() const override {
    return "Trans Layout from cl::Buffer(NCHW) to cl::Image2D(RGBA)";
  }

 private:
  std::string kernel_func_name_{"buffer_to_image2d"};
  std::string build_options_{"-DCL_DTYPE=float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

class LayoutComputeImage2DHwcToBufferChw
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "buffer/layout_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = Param<param_t>();
    auto* y_data = param.y->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    auto y_dims = param.y->dims();
    auto* x_data = param.x->data<float, cl::Image2D>();
    auto x_dims = param.x->dims();

    std::vector<size_t> new_dims = {1, 1, 1, 1};
    for (int j = 0; j < x_dims.size(); ++j) {
      new_dims[4 - x_dims.size() + j] = x_dims[j];
    }

    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];
    VLOG(4) << "new_dims[" << new_dims.size() << "D]:" << new_dims[0] << " "
            << new_dims[1] << " " << new_dims[2] << " " << new_dims[3];

    size_t C = new_dims[1];
    size_t in_height = new_dims[2];
    size_t in_width = new_dims[3];
    int size_ch = in_height * in_width;
    int size_block = size_ch * 4;
    int size_batch = size_ch * C;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_width));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_height));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(size_ch));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(size_ch));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(size_batch));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(C));
    CL_CHECK_FATAL(status);
    VLOG(4) << "gws:[3D]" << ((new_dims[1] + 3) / 4) << " " << new_dims[3]
            << " " << (new_dims[0] * new_dims[2]);
    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>((new_dims[1] + 3) / 4),
                    static_cast<cl::size_type>(new_dims[3]),
                    static_cast<cl::size_type>(new_dims[0] * new_dims[2])};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    // TODO(ysh329): io_copy(device->host) jammed if emplace to `cl_wait_list`
    // context.cl_wait_list()->emplace(y_data, event_);
    context.cl_context()->GetCommandQueue().finish();
  }

  std::string doc() const override {
    return "Trans Layout from cl::Image2D(RGBA) to cl::Buffer(NCHW)";
  }

 private:
  std::string kernel_func_name_{"image2d_to_buffer"};
  std::string build_options_{"-DCL_DTYPE=float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// BufferChwToImage2DHwc
// [chw] -> [hwc]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kFloat,
    kNHWC,
    paddle::lite::kernels::opencl::LayoutComputeBufferChwToImage2DHwc,
    buffer_chw_to_image2d_hwc_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

// [chw] -> [hwc]
REGISTER_LITE_KERNEL(
    layout_once,
    kOpenCL,
    kFloat,
    kNHWC,
    paddle::lite::kernels::opencl::LayoutComputeBufferChwToImage2DHwc,
    buffer_chw_to_image2d_hwc_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

// Image2DHwcBufferChw
// [hwc] -> [chw]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kFloat,
    kNCHW,
    paddle::lite::kernels::opencl::LayoutComputeImage2DHwcToBufferChw,
    image2d_hwc_to_buffer_chw_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

// [hwc] -> [chw]
REGISTER_LITE_KERNEL(
    layout_once,
    kOpenCL,
    kFloat,
    kNCHW,
    paddle::lite::kernels::opencl::LayoutComputeImage2DHwcToBufferChw,
    image2d_hwc_to_buffer_chw_opencl_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
