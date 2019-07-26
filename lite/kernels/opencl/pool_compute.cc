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
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/opencl/cl_include.h"
#include "lite/operators/op_params.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class PoolCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::PoolParam;

  void Run() override {
    const auto& param = *param_.get_mutable<param_t>();
    const auto& in_dims = param.x->dims();
    const auto& out_dims = param.output->dims();
    const std::string pooling_type = param.pooling_type;
    const bool global_pooling = param.global_pooling;
    std::vector<int> paddings = param.paddings;
    std::vector<int> strides = param.strides;
    std::vector<int> ksize = param.ksize;
    if (global_pooling) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_dims[i + 2]);
      }
    }

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* input_buf = param.x->data<float, cl::Buffer>();
    auto* output_buf =
        param.output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    auto kernel = context.cl_context()->GetKernel(
        string_format("pool_%s", pooling_type.c_str()));
    cl_int status;
    auto numel = out_dims.production();
    status = kernel.setArg(0, static_cast<const int>(numel));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *input_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, static_cast<const int>(in_dims[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, static_cast<const int>(in_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, static_cast<const int>(in_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, static_cast<const int>(out_dims[2]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(6, static_cast<const int>(out_dims[3]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(7, static_cast<const int>(ksize[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(8, static_cast<const int>(ksize[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(9, static_cast<const int>(strides[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(10, static_cast<const int>(strides[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(11, static_cast<const int>(paddings[0]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(12, static_cast<const int>(paddings[1]));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(13, *output_buf);
    CL_CHECK_FATAL(status);
    cl::Event event;
    auto global_work_size = cl::NDRange(static_cast<size_t>(numel));
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

REGISTER_LITE_KERNEL(pool2d,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::PoolCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
