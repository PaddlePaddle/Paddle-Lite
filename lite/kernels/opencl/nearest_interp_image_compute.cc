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

#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class NearestInterpComputeImageDefault
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::InterpolateParam;

  std::string doc() const override {
    return "NearestInterp using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/nearest_interp_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    const auto& y_dims = param.Out->dims();
    auto* x_img = GET_DATA_GPU(param.X);
    auto out_image_shape = InitImageDimInfoWith(y_dims);
    auto* out_img = MUTABLE_DATA_GPU(param.Out,
                                     out_image_shape["width"],
                                     out_image_shape["height"],
                                     nullptr);
    float align_data = (param.align_corners) ? 0.5 : 0.0f;
    float ratio_h = 0.f;
    float ratio_w = 0.f;

    if (param.version_2) {
      auto SizeTensor = param.SizeTensor;
      auto OutSize = param.OutSize;
      auto Scale = param.Scale;
      float scale_h = -1.f;
      float scale_w = -1.f;
      if (!SizeTensor.empty()) {
      } else if (OutSize) {
      } else {
        if (Scale) {
          scale_h = Scale->data<float>()[0];
          scale_w = Scale->data<float>()[1];
        } else {
          scale_h = param.scale_v[0];
          scale_w = param.scale_v[1];
        }
      }

      if (y_dims[2] > 1) {
        float new_scale_h = 0.f;
        new_scale_h = (scale_h > 0) ? static_cast<float>(1. / scale_h)
                                    : static_cast<float>(x_dims[2]) / y_dims[2];
        ratio_h = (param.align_corners)
                      ? static_cast<float>(x_dims[2] - 1) / (y_dims[2] - 1)
                      : static_cast<float>(new_scale_h);
      }
      if (y_dims[3] > 1) {
        float new_scale_w = 0.f;
        new_scale_w = (scale_w > 0) ? static_cast<float>(1. / scale_w)
                                    : static_cast<float>(x_dims[3]) / y_dims[3];
        ratio_w = (param.align_corners)
                      ? static_cast<float>(x_dims[3] - 1) / (y_dims[3] - 1)
                      : static_cast<float>(new_scale_w);
      }
    } else {
      if (y_dims[2] > 1) {
        ratio_h = (param.align_corners)
                      ? static_cast<float>(x_dims[2] - 1) / (y_dims[2] - 1)
                      : static_cast<float>(x_dims[2]) / y_dims[2];
      }
      if (y_dims[3] > 1) {
        ratio_w = (param.align_corners)
                      ? static_cast<float>(x_dims[3] - 1) / (y_dims[3] - 1)
                      : static_cast<float>(x_dims[3]) / y_dims[3];
      }
    }
    if (y_dims[3] == x_dims[3] && y_dims[2] == x_dims[2]) {
      ratio_h = 1.f;
      ratio_w = 1.f;
    }

    int in_dims_h = x_dims[2];
    int out_dims_h = y_dims[2];
    int in_dims_w = x_dims[3];
    int out_dims_w = y_dims[3];

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(ratio_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(ratio_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(align_data));
    CL_CHECK_FATAL(status);

#ifdef LITE_WITH_LOG
    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Out->target());
    VLOG(4) << "out_image_shape(w,h):" << out_image_shape["width"] << " "
            << out_image_shape["height"];
    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];
#endif

    const std::vector<size_t>& default_work_size = DefaultGlobalWorkSize(
        y_dims,
        DDim(std::vector<DDim::value_type>{
            static_cast<int64_t>(out_image_shape["width"]),
            static_cast<int64_t>(out_image_shape["height"])}));
    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(default_work_size.data()[0]),
                    static_cast<cl::size_type>(default_work_size.data()[1]),
                    static_cast<cl::size_type>(default_work_size.data()[2])};

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  std::string kernel_func_name_{"nearest_interp"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    nearest_interp,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::NearestInterpComputeImageDefault,
    ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp_v2,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::NearestInterpComputeImageDefault,
    ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
