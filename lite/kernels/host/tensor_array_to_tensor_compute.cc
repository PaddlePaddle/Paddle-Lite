// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/tensor_array_to_tensor_compute.h"
#include <vector>
#include "lite/backends/host/math/concat.h"
#include "lite/backends/host/math/stack.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void TensorArrayToTensorCompute::Run() {
  auto& param = this->Param<param_t>();
  auto OutIndex = param.OutIndex;
  auto X = *param.X;
  int axis = param.axis;
  size_t n = X.size();
  auto OutIndex_data = OutIndex->mutable_data<int32_t>();

  std::vector<Tensor*> inputs;
  for (int i = 0; i < n; i++) {
    auto& input_dims_i = X[i].dims();
    OutIndex_data[i] = input_dims_i[axis];
    inputs.push_back(&X[i]);
  }

  bool use_stack = param.use_stack;
  auto out = param.Out;
  if (inputs.size() > 0) {
    auto precision = inputs[0]->precision();
    if (use_stack) {
      switch (precision) {
        case PRECISION(kFloat):
          lite::host::math::stack_func<float>(inputs, axis, out);
          break;
        case PRECISION(kInt64):
          lite::host::math::stack_func<int64_t>(inputs, axis, out);
          break;
        default:
          LOG(ERROR) << "Unsupported type " << PrecisionToStr(precision);
          break;
      }
    } else {
      switch (precision) {
        case PRECISION(kFloat):
          lite::host::math::concat_func<float>(inputs, axis, out);
          break;
        case PRECISION(kInt64):
          lite::host::math::concat_func<int64_t>(inputs, axis, out);
          break;
        default:
          LOG(ERROR) << "Unsupported type " << PrecisionToStr(precision);
          break;
      }
    }
  }
  param.X->clear();
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(tensor_array_to_tensor,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::TensorArrayToTensorCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorListTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("OutIndex",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
