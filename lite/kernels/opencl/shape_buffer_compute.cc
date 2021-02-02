// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/opencl/cl_utility.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

extern float CopyFromHostSync(void* target, const void* source, size_t size);

class ShapeCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  using param_t = operators::ShapeParam;

  std::string doc() const override { return "Shape using cl::Buffer, kAny"; }

  void Run() {
    auto& param = Param<operators::ShapeParam>();
    auto in_dims = param.X->dims();
    std::vector<int> output_cpu(in_dims.size());
    for (int i = 0; i < in_dims.size(); ++i) {
      output_cpu[i] = in_dims[i];
    }

    auto* data = param.Out->mutable_data<int, cl::Buffer>(TARGET(kOpenCL));
    h2d_duration_ =
        CopyFromHostSync(data, output_cpu.data(), param.Out->memory_size());
#ifdef LITE_WITH_LOG
    size_t buffer_size;
    data->getInfo(CL_MEM_SIZE, &buffer_size);
    VLOG(4) << "out of shape, opencl buffer size: " << buffer_size;
    VLOG(4) << "shape out dims: " << param.Out->dims();
    VLOG(4) << "param.Out->memory_size():" << param.Out->memory_size();
#endif
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = "shape";
    ch->io_duration = h2d_duration_;
  }
#endif

  float h2d_duration_{0};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(shape,
                     kOpenCL,
                     kAny,
                     kAny,
                     paddle::lite::kernels::opencl::ShapeCompute,
                     def)
    .BindInput("Input",
               {LiteType::GetTensorTy(
                   TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kOpenCL), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .Finalize();
