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
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/logging.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {
class Pad2dCompute : public KernelLite<TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::Pad2dParam;

  std::string doc() const override {
    return "Pad2d using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    pad2d_param_ = param_.get_mutable<param_t>();

    if (pad2d_param_->mode == "constant") {
      kernel_func_name_ = "pad2d_constant";
    } else if (pad2d_param_->mode == "reflect") {
      kernel_func_name_ = "pad2d_reflect";
    } else if (pad2d_param_->mode == "edge") {
      kernel_func_name_ = "pad2d_edge";
    } else {
      LOG(FATAL) << "Unknown mode type";
    }

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/pad2d_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto* x = pad2d_param_->X;
    auto* out = pad2d_param_->Out;
    auto out_dims = out->dims();
    auto in_dims = x->dims();

    int in_h = in_dims[2];
    int in_w = in_dims[3];
    int out_h = out_dims[2];
    int out_w = out_dims[3];

#ifdef LITE_WITH_LOG
    VLOG(4) << "x->target():" << TargetToStr(x->target());
    VLOG(4) << "out->target():" << TargetToStr(out->target());
    VLOG(4) << "x->dims():" << in_dims;
    VLOG(4) << "out->dims():" << out_dims;
#endif

    auto out_image_shape = InitImageDimInfoWith(out_dims);
    auto* x_img = x->data<half_t, cl::Image2D>();

    auto* out_img = out->mutable_data<half_t, cl::Image2D>(
        out_image_shape["width"], out_image_shape["height"]);

#ifdef LITE_WITH_LOG
    VLOG(4) << "out_image_shape[w,h]: " << out_image_shape["width"] << " "
            << out_image_shape["height"];

    VLOG(4) << "in_h: " << in_h << ", in_w: " << in_w;
    VLOG(4) << "out_h: " << out_h << ", out_w: " << out_w;
#endif

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    auto default_work_size = DefaultGlobalWorkSize(
        out_dims,
        DDim(std::vector<DDim::value_type>{
            static_cast<int64_t>(out_image_shape["width"]),
            static_cast<int64_t>(out_image_shape["height"])}));
#ifdef LITE_WITH_LOG
    VLOG(4) << "default_work_size: " << default_work_size[0] << ", "
            << default_work_size[1] << ", " << default_work_size[2];
#endif
    int pad_h0 = pad2d_param_->paddings[0];
    int pad_h1 = pad2d_param_->paddings[1];
    int pad_w0 = pad2d_param_->paddings[2];
    int pad_w1 = pad2d_param_->paddings[3];
    float pad_value = pad2d_param_->pad_value;

    cl_int status = kernel.setArg(arg_idx++, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, in_h);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, in_w);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, out_h);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, out_w);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, pad_h0);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, pad_h1);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, pad_w0);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, pad_w1);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, pad_value);
    CL_CHECK_FATAL(status);

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(default_work_size[0]),
                    static_cast<cl::size_type>(default_work_size[1]),
                    static_cast<cl::size_type>(default_work_size[2])};

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
#ifdef LITE_WITH_LOG
    VLOG(4) << "global_work_size:[2D]:" << global_work_size[0] << " "
            << global_work_size[1] << " " << global_work_size[2];
#endif
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 protected:
  param_t* pad2d_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{"-DCL_DTYPE_half"};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(
    pad2d, kOpenCL, kFP16, kImageDefault, ocl::Pad2dCompute, ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
