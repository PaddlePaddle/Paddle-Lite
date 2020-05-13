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
#include "lite/backends/opencl/cl_half.h"
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

// [NCHW] -> [ImageDefault]
class LayoutComputeBufferChwToImageDefault
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kAny),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& param = Param<param_t>();
    if (param.process_type == 1) {
      kernel_func_name_ = "buffer_to_image2d_with_pre255";
    }
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/layout_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    auto& param = Param<param_t>();
    const cl::Buffer* x_data;
    if (param.process_type == 1) {
      x_data = param.x->data<uint8_t, cl::Buffer>();
    } else {
      x_data = param.x->data<float, cl::Buffer>();
    }
    auto x_dims = param.x->dims();
    auto image_shape = InitImageDimInfoWith(x_dims);
    auto* y_data = param.y->mutable_data<half_t, cl::Image2D>(
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

#ifdef LITE_WITH_LOG
    VLOG(2) << "param.process_type:" << param.process_type;
    VLOG(2) << "x_dims:" << x_dims;
    VLOG(2) << "param.x->memory_size():" << param.x->memory_size();
    VLOG(2) << "new_dims[" << new_dims.size() << "D]:" << new_dims[0] << " "
            << new_dims[1] << " " << new_dims[2] << " " << new_dims[3];
    VLOG(2) << "y_dims:" << y_dims;
    VLOG(2) << "param.y->memory_size():" << param.y->memory_size();
    VLOG(2) << "y image_shape(w,h):" << image_shape["width"] << " "
            << image_shape["height"];
    VLOG(2) << "out_C:" << out_C;
    VLOG(2) << "out_H:" << out_H;
    VLOG(2) << "out_W:" << out_W;
    VLOG(2) << "Stride2:" << Stride2;
    VLOG(2) << "Stride1:" << Stride1;
    VLOG(2) << "Stride0:" << Stride0;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
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

#ifdef LITE_WITH_LOG
    VLOG(2) << "gws:[3D]" << ((new_dims[1] + 3) / 4) << " " << new_dims[3]
            << " " << (new_dims[0] * new_dims[2]);
#endif

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
        nullptr);
    CL_CHECK_FATAL(status);
  }

  std::string doc() const override {
    return "Trans Layout from cl::Buffer(NCHW) to "
           "cl::Image2D(ImageDefault/RGBA), Float ---> FP16";
  }

 private:
  std::string time_stamp_{GetTimeStamp()};
  std::string kernel_func_name_{"buffer_to_image2d"};
  std::string build_options_{"-DCL_DTYPE_float"};
};

// [ImageDefault] -> [NCHW]
class LayoutComputeImageDefaultToBufferChw
    : public KernelLite<TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& param = Param<param_t>();
    if (param.process_type == 1) {
      kernel_func_name_ = "image2d_to_buffer_with_post255";
    }
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/layout_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    auto& param = Param<param_t>();
    const cl::Buffer* y_data;
    if (param.process_type == 1) {
      y_data = param.y->mutable_data<uint8_t, cl::Buffer>(TARGET(kOpenCL));
    } else {
      y_data = param.y->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    }
    auto* x_data = param.x->data<half_t, cl::Image2D>();
    auto x_dims = param.x->dims();
    auto y_dims = param.y->dims();
    auto x_image_shape = InitImageDimInfoWith(x_dims);

    std::vector<size_t> new_dims = {1, 1, 1, 1};
    for (int j = 0; j < x_dims.size(); ++j) {
      new_dims[4 - x_dims.size() + j] = x_dims[j];
    }

#ifdef LITE_WITH_LOG
    VLOG(2) << "param.process_type:" << param.process_type;
    VLOG(2) << "x_dims:" << x_dims;
    VLOG(2) << "param.x->memory_size():" << param.x->memory_size();
    VLOG(2) << "x_image_shape(w,h):" << x_image_shape["width"] << " "
            << x_image_shape["height"];
    VLOG(2) << "new_dims[" << new_dims.size() << "D]:" << new_dims[0] << " "
            << new_dims[1] << " " << new_dims[2] << " " << new_dims[3];
    VLOG(2) << "y_dims:" << y_dims;
    VLOG(2) << "param.y->memory_size():" << param.y->memory_size();
#endif

    size_t C = new_dims[1];
    size_t in_height = new_dims[2];
    size_t in_width = new_dims[3];
    int size_ch = in_height * in_width;
    int size_block = size_ch * 4;
    int size_batch = size_ch * C;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
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
    status = kernel.setArg(++arg_idx, static_cast<const int>(size_block));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(size_batch));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(C));
    CL_CHECK_FATAL(status);
#ifdef LITE_WITH_LOG
    VLOG(2) << "gws:[3D]" << ((new_dims[1] + 3) / 4) << " " << new_dims[3]
            << " " << (new_dims[0] * new_dims[2]);
#endif
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
        nullptr);
    CL_CHECK_FATAL(status);
  }

  std::string doc() const override {
    return "Trans Layout from cl::Image2D(ImageDefault/RGBA) to "
           "cl::Buffer(NCHW), FP16 ---> Float";
  }

 private:
  std::string time_stamp_{GetTimeStamp()};
  std::string kernel_func_name_{"image2d_to_buffer"};
  std::string build_options_{"-DCL_DTYPE_float"};
};

// [NCHW] -> [ImageDW]
class LayoutComputeBufferChwToImage2DNw
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFloat),
                        DATALAYOUT(kImageNW)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "buffer/layout_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    auto& param = Param<param_t>();
    auto* x_data = param.x->data<float, cl::Buffer>();
    auto x_dims = param.x->dims();

    CHECK(x_dims.size() == 4) << " Tensor dim is not 4.";
    size_t image_width = x_dims[3] * ((x_dims[0] + 3) / 4);
    size_t image_height = x_dims[1] * x_dims[2];

    auto* y_data =
        param.y->mutable_data<float, cl::Image2D>(image_width, image_height);
    auto y_dims = param.y->dims();

    // out info
    std::vector<size_t> new_dims = {1, 1, 1, 1};
    for (int tidx = 0; tidx < x_dims.size(); ++tidx) {
      new_dims[4 - x_dims.size() + tidx] = x_dims[tidx];
    }

    const int out_N = new_dims[0];
    const int out_C = new_dims[1];
    const int out_H = new_dims[2];
    const int out_W = new_dims[3];

    const int Stride2 = out_C * out_H * out_W;
    const int Stride1 = out_H * out_W;
    const int Stride0 = out_W;

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
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
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_N));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(Stride0));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(Stride1));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(Stride2));
    CL_CHECK_FATAL(status);

    VLOG(2) << "gws:[3D]" << ((out_N + 3) / 4) << " " << out_W << " "
            << (out_C * out_H);
    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>((out_N + 3) / 4),  // N blocks
                    static_cast<cl::size_type>(out_W),            // w
                    static_cast<cl::size_type>(out_C * out_H)};   // ch

    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        nullptr);
    CL_CHECK_FATAL(status);
  }

  std::string doc() const override {
    return "Trans Layout from cl::Buffer(NCHW) to cl::Image2D(ImageDW/CLNW)";
  }

 private:
  std::string time_stamp_{GetTimeStamp()};

  std::string kernel_func_name_{"buffer_to_image2d_nw"};
  std::string build_options_{"-DCL_DTYPE_float "};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// [NCHW] -> [ImageDefault]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kAny,
    kImageDefault,
    paddle::lite::kernels::opencl::LayoutComputeBufferChwToImageDefault,
    NCHW_to_ImageDefault)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

// [ImageDefault] -> [NCHW]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kAny,
    kNCHW,
    paddle::lite::kernels::opencl::LayoutComputeImageDefaultToBufferChw,
    ImageDefault_to_NCHW)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
