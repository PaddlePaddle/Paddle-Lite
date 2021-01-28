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

#include "lite/kernels/host/gather_tree_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void GatherTreeCompute<T>::Run() {
  auto& param = this->template Param<operators::GatherTreeParam>();
  const auto* ids_data = param.ids->template data<T>();
  const auto* parents_data = param.parents->template data<T>();
  auto* out_data = param.out->template mutable_data<T>();
  auto& ids_dims = param.ids->dims();
  int max_length = ids_dims[0];
  int batch_size = ids_dims[1];
  int beam_size = ids_dims[2];

  for (int batch = 0; batch < batch_size; batch++) {
    for (int beam = 0; beam < beam_size; beam++) {
      auto idx =
          (max_length - 1) * batch_size * beam_size + batch * beam_size + beam;
      out_data[idx] = ids_data[idx];
      auto parent = parents_data[idx];
      for (int step = max_length - 2; step >= 0; step--) {
        idx = step * batch_size * beam_size + batch * beam_size;
        out_data[idx + beam] = ids_data[idx + parent];
        parent = parents_data[idx + parent];
      }
    }
  }
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using GatherTreeInt32 = paddle::lite::kernels::host::GatherTreeCompute<int32_t>;
REGISTER_LITE_KERNEL(gather_tree, kHost, kFloat, kAny, GatherTreeInt32, int32)
    .BindInput("Ids",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Parents",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using GatherTreeInt64 = paddle::lite::kernels::host::GatherTreeCompute<int64_t>;
REGISTER_LITE_KERNEL(gather_tree, kHost, kFloat, kAny, GatherTreeInt64, int64)
    .BindInput("Ids",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Parents",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
