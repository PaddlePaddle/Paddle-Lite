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

#include "lite/kernels/host/assign_value_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void TensorFromVector(const std::vector<T>& src, lite::Tensor* dst) {
  auto* src_ptr = static_cast<const void*>(src.data());
  auto* dst_ptr = static_cast<void*>(dst->mutable_data<T>());
  auto size = src.size() * sizeof(T);
  lite::TargetWrapperHost::MemcpySync(
      dst_ptr, src_ptr, size, IoDirection::HtoH);
}

void AssignValueCompute::Run() {
  auto& param = Param<operators::AssignValueParam>();
  int dtype = param.dtype;
  std::vector<float> fp32_values = param.fp32_values;
  std::vector<int> int32_values = param.int32_values;
  std::vector<int64_t> int64_values = param.int64_values;
  std::vector<int> bool_values = param.bool_values;
  auto* out = param.Out;

  if (dtype == static_cast<int>(lite::core::FluidType::INT32)) {
    TensorFromVector(int32_values, out);
  } else if (dtype == static_cast<int>(lite::core::FluidType::FP32)) {
    TensorFromVector(fp32_values, out);
  } else if (dtype == static_cast<int>(lite::core::FluidType::INT64)) {
    TensorFromVector(int64_values, out);
  } else if (dtype == static_cast<int>(lite::core::FluidType::BOOL)) {
    TensorFromVector(bool_values, out);
  } else {
    LOG(FATAL) << "Unsupported dtype for assign_value_op:" << dtype;
  }
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(assign_value,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::AssignValueCompute,
                     def)
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
