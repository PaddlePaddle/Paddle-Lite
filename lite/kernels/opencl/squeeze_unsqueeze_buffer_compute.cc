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

#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

extern float CopyFromDeviceToDeviceSync(void* target,
                                        const void* source,
                                        size_t size);

class SqueezeUnsqueezeCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::SqueezeParam;

  std::string doc() const override {
    return "squeeze using cl::Buffer, kFloat";
  }
  void Run() override {
    const Tensor* x;
    Tensor* out;

    // choose kernel_param according to op param_
    if (param_.is_type<param_t>()) {
      auto* kernel_param = param_.get_mutable<param_t>();
      x = kernel_param->X;
      out = kernel_param->Out;
      kernel_inplace = kernel_param->inplace;
    } else {
      auto* kernel_param = param_.get_mutable<operators::UnsqueezeParam>();
      x = kernel_param->X;
      out = kernel_param->Out;
      kernel_inplace = kernel_param->inplace;
    }

    auto* x_data = x->data<float, cl::Buffer>();
    if (kernel_inplace) {
      auto out_dims = out->dims();
      out->ShareDataWith(*x);
      out->Resize(out_dims);
    } else {
      auto* out_data = out->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
      d2d_duration_ =
          CopyFromDeviceToDeviceSync(out_data, x_data, out->memory_size());
    }
#ifdef LITE_WITH_LOG
    size_t buffer_size;
    x_data->getInfo(CL_MEM_SIZE, &buffer_size);
    VLOG(4) << "out of squeeze, opencl buffer size: " << buffer_size;
    VLOG(4) << "squeeze out dims: " << out->dims();
    VLOG(4) << "out->memory_size():" << out->memory_size();
#endif
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    if (!kernel_inplace) {
      ch->kernel_func_name = "io_copy_d2d";
      ch->io_duration = d2d_duration_;
    }
  }
#endif

  float d2d_duration_{0};
  bool kernel_inplace;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(squeeze,
                     kOpenCL,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::opencl::SqueezeUnsqueezeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(squeeze2,
                     kOpenCL,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::opencl::SqueezeUnsqueezeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(unsqueeze,
                     kOpenCL,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::opencl::SqueezeUnsqueezeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW))})
    .BindInput("AxesTensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kNCHW))})
    .BindInput("AxesTensorList",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(unsqueeze2,
                     kOpenCL,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::opencl::SqueezeUnsqueezeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW))})
    .BindInput("AxesTensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("AxesTensorList",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
