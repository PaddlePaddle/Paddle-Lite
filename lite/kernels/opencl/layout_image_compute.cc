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
#include "lite/backends/opencl/cl_utility.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/type_system.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/log/cp_logging.h"

#undef LITE_WITH_LOG

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
    if (!fp16_support_) {
      build_options_ += " -DCL_DTYPE_FLOAT_FORCE";
    }
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/layout_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

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
    auto* y_data = MUTABLE_DATA_GPU(
        param.y, image_shape["width"], image_shape["height"], nullptr);
    auto y_dims = param.y->dims();

    // out info
    std::vector<size_t> new_dims = {1, 1, 1, 1};
    if (x_dims.size() == 5) {
      new_dims[4 - x_dims.size() + 1] = x_dims[0] * x_dims[1];
      for (int tidx = 2; tidx < x_dims.size(); ++tidx) {
        new_dims[4 - x_dims.size() + tidx] = x_dims[tidx];
      }
    } else if (x_dims.size() < 5) {
      for (int tidx = 0; tidx < x_dims.size(); ++tidx) {
        new_dims[4 - x_dims.size() + tidx] = x_dims[tidx];
      }
    } else {
      LOG(FATAL) << "unsupported layout tensor dims size, the dims size is:"
                 << x_dims.size();
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

    cl_int status;
    status = kernel.setArg(0, *x_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *y_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, static_cast<const int>(out_H));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, static_cast<const int>(out_W));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, static_cast<const int>(out_C));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, static_cast<const int>(Stride0));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(6, static_cast<const int>(Stride1));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(7, static_cast<const int>(Stride2));
    CL_CHECK_FATAL(status);

#ifdef LITE_WITH_LOG
    VLOG(2) << "gws:[3D]" << ((new_dims[1] + 3) / 4) << " " << new_dims[3]
            << " " << (new_dims[0] * new_dims[2]);
#endif

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>((new_dims[1] + 3) / 4),
                    static_cast<cl::size_type>(new_dims[3]),
                    static_cast<cl::size_type>(new_dims[0] * new_dims[2])};

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
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
    if (!fp16_support_) {
      build_options_ += " -DCL_DTYPE_FLOAT_FORCE";
    }
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/layout_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void Run() override {
    auto& param = Param<param_t>();
    const cl::Buffer* y_data;
    if (param.process_type == 1) {
      y_data = param.y->mutable_data<uint8_t, cl::Buffer>(TARGET(kOpenCL));
    } else {
      y_data = param.y->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    }
    auto* x_data = GET_DATA_GPU(param.x);
    auto x_dims = param.x->dims();
    auto y_dims = param.y->dims();
    auto x_image_shape = InitImageDimInfoWith(x_dims);

    std::vector<size_t> new_dims = {1, 1, 1, 1};
    if (x_dims.size() == 5) {
      new_dims[4 - x_dims.size() + 1] = x_dims[0] * x_dims[1];
      for (int j = 2; j < x_dims.size(); ++j) {
        new_dims[4 - x_dims.size() + j] = x_dims[j];
      }
    } else if (x_dims.size() < 5) {
      for (int j = 0; j < x_dims.size(); ++j) {
        new_dims[4 - x_dims.size() + j] = x_dims[j];
      }
    } else {
      LOG(FATAL) << "unsupported layout tensor dims size, the dims size is: "
                 << x_dims.size();
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

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
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

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

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
    if (x_dims.size() == 5) {
      new_dims[4 - x_dims.size() + 1] = x_dims[0] * x_dims[1];
      for (int tidx = 2; tidx < x_dims.size(); ++tidx) {
        new_dims[4 - x_dims.size() + tidx] = x_dims[tidx];
      }
    } else if (x_dims.size() < 5) {
      for (int tidx = 0; tidx < x_dims.size(); ++tidx) {
        new_dims[4 - x_dims.size() + tidx] = x_dims[tidx];
      }
    } else {
      LOG(FATAL) << "unsupported layout tensor dims size, the dims size is:"
                 << x_dims.size();
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

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
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

// [ImageDefault] -> [ImageFolder]
class LayoutComputeImageDefaultToImageFolder
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kAny),
                        DATALAYOUT(kImageFolder)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& param = Param<param_t>();
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/layout_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void Run() override {
    auto& param = Param<param_t>();
    auto x_dims = param.x->dims();
    auto y_dims = param.y->dims();

    CLImageConverterDefault default_converter;
    CLImageConverterFolder folder_converter;
    auto x_image_shape = default_converter.InitImageDimInfoWith(x_dims);
    auto y_image_shape = folder_converter.InitImageDimInfoWith(y_dims);

    const cl::Image2D* y_data =
        MUTABLE_DATA_GPU(param.y, y_image_shape[0], y_image_shape[1], nullptr);
    auto* x_data = GET_DATA_GPU(param.x);

#ifdef LITE_WITH_LOG
    VLOG(2) << "x_dims:" << x_dims;
    VLOG(2) << "y_dims:" << y_dims;
    VLOG(2) << "x_image_shape(w,h):" << x_image_shape[0] << " "
            << x_image_shape[1];
    VLOG(2) << "y_image_shape(w,h):" << y_image_shape[0] << " "
            << y_image_shape[1];
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status;
    status = kernel.setArg(arg_idx, *x_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_image_shape[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(x_image_shape[1]));
    CL_CHECK_FATAL(status);

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(y_image_shape[0]),
                    static_cast<cl::size_type>(y_image_shape[1])};
#ifdef LITE_WITH_LOG
    for (auto i = 0; i < global_work_size.dimensions(); i++) {
      VLOG(2) << "global_work_size[" << i << "]: " << global_work_size[i];
    }
#endif

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

  std::string doc() const override {
    return "Trans Layout from cl::Image2D(ImageDefault/RGBA) to "
           "cl::Image2D(ImageFolder)";
  }

 private:
  std::string time_stamp_{GetTimeStamp()};
  std::string kernel_func_name_{"image2d_default_to_image2d_folder"};
  std::string build_options_{""};
};

// [ImageFolder] -> [ImageDefault]
class LayoutComputeImageFolderToImageDefault
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kAny),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& param = Param<param_t>();
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/layout_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void Run() override {
    auto& param = Param<param_t>();
    auto x_dims = param.x->dims();
    auto y_dims = param.y->dims();

    CLImageConverterFolder folder_converter;
    CLImageConverterDefault default_converter;
    auto x_image_shape = folder_converter.InitImageDimInfoWith(x_dims);
    auto y_image_shape = default_converter.InitImageDimInfoWith(y_dims);

    const cl::Image2D* y_data =
        MUTABLE_DATA_GPU(param.y, y_image_shape[0], y_image_shape[1], nullptr);
    auto* x_data = GET_DATA_GPU(param.x);

#ifdef LITE_WITH_LOG
    VLOG(2) << "x_dims:" << x_dims;
    VLOG(2) << "y_dims:" << y_dims;
    VLOG(2) << "x_image_shape(w,h):" << x_image_shape[0] << " "
            << x_image_shape[1];
    VLOG(2) << "y_image_shape(w,h):" << y_image_shape[0] << " "
            << y_image_shape[1];
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status;
    status = kernel.setArg(arg_idx, *x_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(y_image_shape[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(y_image_shape[1]));
    CL_CHECK_FATAL(status);

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(x_image_shape[0]),
                    static_cast<cl::size_type>(x_image_shape[1])};
#ifdef LITE_WITH_LOG
    for (auto i = 0; i < global_work_size.dimensions(); i++) {
      VLOG(2) << "global_work_size[" << i << "]: " << global_work_size[i];
    }
#endif

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

  std::string doc() const override {
    return "Trans Layout from cl::Image2D(ImageFolder) to "
           "cl::Image2D(ImageDefault/RGBA)";
  }

 private:
  std::string time_stamp_{GetTimeStamp()};
  std::string kernel_func_name_{"image2d_folder_to_image2d_default"};
  std::string build_options_{""};
};

// [ImageFolder] -> [NCHW]
class LayoutComputeImageFolderToBufferChw
    : public KernelLite<TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& param = Param<param_t>();
    auto x_dims = param.x->dims();
    if (x_dims.size() > 2) {
      kernel_func_name_ = "image2d_to_buffer";
    }
    if (!fp16_support_) {
      build_options_ += " -DCL_DTYPE_FLOAT_FORCE";
    }
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/layout_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void Run() override {
    auto& param = Param<param_t>();
    auto x_dims = param.x->dims();
    auto y_dims = param.y->dims();

    DDim x_image_shape;
    if (x_dims.size() > 2) {
      CLImageConverterDefault default_converter;
      x_image_shape = default_converter.InitImageDimInfoWith(x_dims);
    } else {
      CLImageConverterFolder folder_converter;
      x_image_shape = folder_converter.InitImageDimInfoWith(x_dims);
    }

    const cl::Buffer* y_data =
        param.y->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    auto* x_data = GET_DATA_GPU(param.x);

    // out info
    std::vector<size_t> new_dims = {1, 1, 1, 1};
    for (int tidx = 0; tidx < x_dims.size(); ++tidx) {
      new_dims[4 - x_dims.size() + tidx] = x_dims[tidx];
    }

#ifdef LITE_WITH_LOG
    VLOG(2) << "x_dims:" << x_dims;
    VLOG(2) << "y_dims:" << y_dims;
    VLOG(2) << "x_image_shape(w,h):" << x_image_shape[0] << " "
            << x_image_shape[1];
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    cl::NDRange global_work_size;
    int arg_idx = 0;
    cl_int status;
    if (x_dims.size() <= 2) {
      status = kernel.setArg(arg_idx, *x_data);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, *y_data);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, static_cast<const int>(y_dims[0]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, static_cast<const int>(y_dims[1]));
      CL_CHECK_FATAL(status);

      global_work_size =
          cl::NDRange{static_cast<cl::size_type>(x_image_shape[0]),
                      static_cast<cl::size_type>(x_image_shape[1])};
    } else {
      size_t C = new_dims[1];
      size_t in_height = new_dims[2];
      size_t in_width = new_dims[3];
      int size_ch = in_height * in_width;
      int size_block = size_ch * 4;
      int size_batch = size_ch * C;

      status = kernel.setArg(arg_idx, *x_data);
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

      global_work_size =
          cl::NDRange{static_cast<cl::size_type>((new_dims[1] + 3) / 4),
                      static_cast<cl::size_type>(new_dims[3]),
                      static_cast<cl::size_type>(new_dims[0] * new_dims[2])};
    }

#ifdef LITE_WITH_LOG
    for (auto i = 0; i < global_work_size.dimensions(); i++) {
      VLOG(2) << "global_work_size[" << i << "]: " << global_work_size[i];
    }
#endif

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

  std::string doc() const override {
    return "Trans Layout from cl::Image2D(ImageFolder) to "
           "cl::Buffer(NCHW)";
  }

 private:
  std::string time_stamp_{GetTimeStamp()};
  std::string kernel_func_name_{"image2d_folder_to_buffer"};
  std::string build_options_{"-DCL_DTYPE_float "};
};

// [NCHW] -> [ImageFolder]
class LayoutComputeBufferChwToImageFolder
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kAny),
                        DATALAYOUT(kImageFolder)> {
 public:
  using param_t = operators::LayoutParam;

  void PrepareForRun() override {
    auto& param = Param<param_t>();
    auto x_dims = param.x->dims();
    if (x_dims.size() > 2) {
      kernel_func_name_ = "buffer_to_image2d";
    }
    if (!fp16_support_) {
      build_options_ += " -DCL_DTYPE_FLOAT_FORCE";
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
    auto x_dims = param.x->dims();
    auto y_dims = param.y->dims();
    DDim image_shape;
    if (y_dims.size() > 2) {
      CLImageConverterDefault default_converter;
      image_shape = default_converter.InitImageDimInfoWith(y_dims);
    } else {
      CLImageConverterFolder folder_converter;
      image_shape = folder_converter.InitImageDimInfoWith(y_dims);
    }
    auto* y_data =
        MUTABLE_DATA_GPU(param.y, image_shape[0], image_shape[1], nullptr);
    auto* x_data = GET_BUFFER_GPU(param.x);

    // out info
    std::vector<size_t> new_dims = {1, 1, 1, 1};
    for (int tidx = 0; tidx < x_dims.size(); ++tidx) {
      new_dims[4 - x_dims.size() + tidx] = x_dims[tidx];
    }

#ifdef LITE_WITH_LOG
    VLOG(2) << "x_dims:" << x_dims;
    VLOG(2) << "y_dims:" << y_dims;
    VLOG(2) << "image_shape(w,h):" << image_shape[0] << " " << image_shape[1];
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status;
    status = kernel.setArg(arg_idx, *x_data);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *y_data);
    CL_CHECK_FATAL(status);
    if (y_dims.size() <= 2) {
      const int length = new_dims[0] * new_dims[1] * new_dims[2] * new_dims[3];
      status = kernel.setArg(++arg_idx, static_cast<const int>(y_dims[0]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, static_cast<const int>(y_dims[1]));
      CL_CHECK_FATAL(status);
      status = kernel.setArg(++arg_idx, static_cast<const int>(length));
      CL_CHECK_FATAL(status);
    } else {
      const int out_C = new_dims[1];
      const int out_H = new_dims[2];
      const int out_W = new_dims[3];
      const int Stride2 = out_C * out_H * out_W;
      const int Stride1 = out_H * out_W;
      const int Stride0 = out_W;
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
    }

    if (y_dims.size() <= 2) {
      gws_ = cl::NDRange{static_cast<cl::size_type>(image_shape[0]),
                         static_cast<cl::size_type>(image_shape[1])};
    } else {
      gws_ = cl::NDRange{static_cast<cl::size_type>((new_dims[1] + 3) / 4),
                         static_cast<cl::size_type>(new_dims[3]),
                         static_cast<cl::size_type>(new_dims[0] * new_dims[2])};
    }

    status = EnqueueNDRangeKernel(
        context, kernel, cl::NullRange, gws_, cl::NullRange, nullptr, event_);
    CL_CHECK_FATAL(status);
  }

  std::string doc() const override {
    return "Trans Layout from cl::Buffer(NCHW) to "
           "cl::Image2D(ImageFolder)";
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->global_work_size = ch->NDRangeToStr(gws_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  std::string time_stamp_{GetTimeStamp()};
  std::string kernel_func_name_{"buffer_to_image2d_folder"};
  std::string build_options_{"-DCL_DTYPE_float "};
  cl::NDRange gws_;
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

REGISTER_LITE_KERNEL(
    layout_once,
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

// [NCHW] -> [ImageFolder]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kAny,
    kImageFolder,
    paddle::lite::kernels::opencl::LayoutComputeBufferChwToImageFolder,
    NCHW_to_ImageFolder)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kImageFolder))})
    .Finalize();

REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kAny,
    kNCHW,
    paddle::lite::kernels::opencl::LayoutComputeImageDefaultToBufferChw,
    def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    layout_once,
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

// [ImageDefault] -> [ImageFolder]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kAny,
    kImageFolder,
    paddle::lite::kernels::opencl::LayoutComputeImageDefaultToImageFolder,
    ImageDefault_to_ImageFolder)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kImageFolder))})
    .Finalize();

// [ImageFolder] -> [ImageDefault]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kAny,
    kImageDefault,
    paddle::lite::kernels::opencl::LayoutComputeImageFolderToImageDefault,
    ImageFolder_to_ImageDefault)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageFolder))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

// [ImageFolder] -> [NCHW]
REGISTER_LITE_KERNEL(
    layout,
    kOpenCL,
    kAny,
    kNCHW,
    paddle::lite::kernels::opencl::LayoutComputeImageFolderToBufferChw,
    ImageFolder_to_NCHW)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageFolder))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

#define LITE_WITH_LOG
