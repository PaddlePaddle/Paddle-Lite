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

#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/opencl/cl_include.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ElementwiseAddCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto axis = param.axis;
    const auto& x_dims = param.X->dims();
    const auto& y_dims = param.Y->dims();
    const auto& out_dims = param.Out->dims();
    if (axis < 0) {
      axis = static_cast<int>(x_dims.size() - y_dims.size());
    }
    size_t batch = 1;
    size_t channels = 1;
    size_t num = 1;
    for (int i = 0; i < axis; ++i) {
      batch *= x_dims[i];
    }
    for (int i = 0; i < y_dims.size(); ++i) {
      channels *= y_dims[i];
    }
    for (int i = static_cast<int>(y_dims.size() + axis); i < x_dims.size();
         ++i) {
      num *= x_dims[i];
    }
    VLOG(4) << "axis:" << axis;
    VLOG(4) << "channels:" << channels;
    VLOG(4) << "num:" << num;
    VLOG(4) << "batch:" << batch;
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* x_buf = param.X->data<float, cl::Buffer>();
    auto* y_buf = param.Y->data<float, cl::Buffer>();
    auto* out_buf = param.Out->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    auto kernel = context.cl_context()->GetKernel("elementwise_add");
    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Y->target());
    VLOG(4) << TargetToStr(param.Out->target());
    cl_int status = kernel.setArg(0, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *y_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, (const int)batch);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, (const int)channels);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, (const int)num);
    CL_CHECK_FATAL(status);

    auto global_work_size = cl::NDRange{channels, batch};
    cl::Event event;
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        &event);
    CL_CHECK_FATAL(status);
    status = event.wait();
    CL_CHECK_FATAL(status);
  }
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::ElementwiseAddCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
