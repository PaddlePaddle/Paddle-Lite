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

#include "lite/kernels/xpu/assign_value_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void AssignValueCompute::Run() {
  auto& param = this->template Param<param_t>();
  int dtype = param.dtype;
  std::vector<float> fp32_values = param.fp32_values;
  std::vector<int> int32_values = param.int32_values;
  std::vector<int64_t> int64_values = param.int64_values;
  CHECK_GT(param.shape.size(), 0UL);
  if (dtype == static_cast<int>(lite::core::FluidType::INT32)) {
    auto* out = param.Out->mutable_data<int>(TARGET(kXPU));
    lite::TargetWrapperXPU::MemcpySync(out,
                                       int32_values.data(),
                                       sizeof(int) * int32_values.size(),
                                       IoDirection::HtoD);
  } else if (dtype == static_cast<int>(lite::core::FluidType::FP32)) {
    auto* out = param.Out->mutable_data<float>(TARGET(kXPU));
    lite::TargetWrapperXPU::MemcpySync(out,
                                       fp32_values.data(),
                                       sizeof(float) * fp32_values.size(),
                                       IoDirection::HtoD);
  } else if (dtype == static_cast<int>(lite::core::FluidType::INT64)) {
    auto* out = param.Out->mutable_data<int64_t>(TARGET(kXPU));
    lite::TargetWrapperXPU::MemcpySync(out,
                                       int64_values.data(),
                                       sizeof(int64_t) * int64_values.size(),
                                       IoDirection::HtoD);
  } else {
    LOG(FATAL) << "Unsupported dtype for assign_value_op:" << dtype;
  }
  return;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(assign_value,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::AssignValueCompute,
                     def)
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
