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
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/elementwise_add_image_compute.h"
#include "lite/kernels/opencl/image_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class FusionElementwiseAddActivationImageCompute
    : public ElementwiseAddImageCompute {
 public:
  using param_t = operators::FusionElementwiseActivationParam;

  void PrepareForRun() override {
    build_options_ += " -DRELU";
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/elementwise_add_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    ele_param_ = param_.get_mutable<param_t>();
    auto act_t = static_cast<param_t*>(ele_param_)->act_type;
    VLOG(4) << "act: " << act_t;
    if (act_t != "relu") {
      LOG(FATAL) << "Unsupported Activation type: " << act_t;
    }
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
  }
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;

REGISTER_LITE_KERNEL(fusion_elementwise_add_activation,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::FusionElementwiseAddActivationImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
