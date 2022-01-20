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
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class DropoutComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                                PRECISION(kFP16),
                                                DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::DropoutParam;

  std::string doc() const override {
    return "Dropout using cl::Image2D, kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/dropout_kernel.cl",
                                    build_options_,
                                    time_stamp_);
  }

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    const auto& in_dims = param.x->dims();
    const auto& out_dims = param.output->dims();
    auto* x_img = GET_DATA_GPU(param.x);
    float dropout_prob;
    if (param.dropout_implementation == "upscale_in_train") {
      dropout_prob = 0.0f;
    } else {
      dropout_prob = param.dropout_prob;
    }

    int input_dims[4] = {1, 1, 1, 1};
    for (int i = 0; i < in_dims.size(); i++) {
      input_dims[4 - in_dims.size() + i] = in_dims[i];
    }
    int out_w = input_dims[3];
    auto out_image_shape = InitImageDimInfoWith(out_dims);
    auto* out_img = MUTABLE_DATA_GPU(param.output,
                                     out_image_shape["width"],
                                     out_image_shape["height"],
                                     nullptr);

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    cl_int status;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, out_w);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, dropout_prob);
    CL_CHECK_FATAL(status);

    const std::vector<size_t>& default_work_size = DefaultGlobalWorkSize(
        out_dims,
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
  std::string kernel_func_name_{"dropout"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(dropout,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::DropoutComputeImage2D,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Seed",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
