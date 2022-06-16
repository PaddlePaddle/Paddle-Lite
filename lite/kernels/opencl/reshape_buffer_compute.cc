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
#include "lite/utils/log/logging.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

#undef LITE_WITH_LOG

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

// reshape operator
class ReshapeComputeFloatBuffer
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ReshapeParam;

  void PrepareForRun() override { auto& context = ctx_->As<OpenCLContext>(); }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    // ch->kernel_func_name = kernel_func_name_;
    // ch->cl_event =
    //     event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const Tensor* const x = param.x;
    Tensor* const output = param.output;

    const auto x_dims = x->dims();
    const DDimLite& out_dims = output->dims();

    VLOG(4) << "in_dims= " << x_dims << "; out_dims= " << out_dims;

    auto* x_buffer = GET_BUFFER_GPU(x);
    auto* y_buffer =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? output->mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL))
            : output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    int size =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? x_dims.production() * sizeof(half_t)
            : x_dims.production() * sizeof(float);
    TargetWrapperCL::MemcpySync(y_buffer, x_buffer, size, IoDirection::DtoD);
    output->Resize(out_dims);
#ifdef LITE_WITH_LOG
    VLOG(4) << TargetToStr(x->target());
    VLOG(4) << TargetToStr(param.output->target());
#endif
  }

 private:
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reshape,
                     kOpenCL,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::opencl::ReshapeComputeFloatBuffer,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2,
                     kOpenCL,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::opencl::ReshapeComputeFloatBuffer,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

#define LITE_WITH_LOG
