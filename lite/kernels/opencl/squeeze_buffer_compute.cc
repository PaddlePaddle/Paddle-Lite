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

class SqueezeCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::SqueezeParam;

  std::string doc() const override {
    return "squeeze using cl::Buffer, kFloat";
  }

  void Run() override {
    param_t* squeeze_param_ = param_.get_mutable<param_t>();
    auto* x_data = squeeze_param_->X->data<float, cl::Buffer>();
    auto* out_data =
        squeeze_param_->Out->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    h2d_duration_ = CopyFromDeviceToDeviceSync(
        out_data, x_data, squeeze_param_->Out->memory_size());
#ifdef LITE_WITH_LOG
    size_t buffer_size;
    out_data->getInfo(CL_MEM_SIZE, &buffer_size);
    VLOG(4) << "out of squeeze, opencl buffer size: " << buffer_size;
    VLOG(4) << "squeeze out dims: " << squeeze_param_->Out->dims();
    VLOG(4) << "squeeze_param_->Out->memory_size():"
            << squeeze_param_->Out->memory_size();
#endif
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = "squeeze";
    ch->io_duration = h2d_duration_;
  }
#endif

  float h2d_duration_{0};
};

class Squeeze2Compute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::SqueezeParam;

  std::string doc() const override {
    return "squeeze2 using cl::Buffer, kFloat";
  }

  void Run() override {
    param_t* squeeze2_param_ = param_.get_mutable<param_t>();
    auto* x_data = squeeze2_param_->X->data<float, cl::Buffer>();
    auto* out_data =
        squeeze2_param_->Out->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    h2d_duration_ = CopyFromDeviceToDeviceSync(
        out_data, x_data, squeeze2_param_->Out->memory_size());
#ifdef LITE_WITH_LOG
    size_t buffer_size;
    out_data->getInfo(CL_MEM_SIZE, &buffer_size);
    VLOG(4) << "out of squeeze2, opencl buffer size: " << buffer_size;
    VLOG(4) << "squeeze2 out dims: " << squeeze2_param_->Out->dims();
    VLOG(4) << "squeeze2_param_->Out->memory_size():"
            << squeeze2_param_->Out->memory_size();
#endif
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = "squeeze2";
    ch->io_duration = h2d_duration_;
  }
#endif

  float h2d_duration_{0};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(squeeze,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::SqueezeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(squeeze2,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::Squeeze2Compute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
