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

#include "lite/kernels/host/scatter_nd_add_compute.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void scatter_nd_add(const int64_t* indexs,
                    const T* updates,
                    T* dst,
                    std::vector<int64_t> in_dims,
                    int index_size,
                    int num,
                    int size);

template <>
void scatter_nd_add<float>(const int64_t* indexs,
                           const float* src,
                           float* dst,
                           std::vector<int64_t> in_dims,
                           int index_size,
                           int num,
                           int size) {
  int64_t offset = num * size;
  for (int i = 0; i < index_size; i++) {
    const int64_t* index_ptr = indexs + i * offset;
    const float* src_ptr = src + i * size;
    for (int j = 0; j < num; j++) {
      const int64_t* index_ptr_n = index_ptr + j * size;
      const float* src_ptr_n = src_ptr + j;
      auto index_data = 0;
      for (int k = 0; k < size; k++) {
        index_data += index_ptr_n[k] * in_dims[k];
      }
      dst[index_data] += src_ptr_n[j];
    }
  }
}

void ScatterNdAddCompute::Run() {
  auto& param = this->template Param<param_t>();
  const float* din_data = param.x->template data<float>();
  const float* updates_data = param.updates->template data<float>();
  const int64_t* indexs_data = param.indexs->template data<int64_t>();
  float* output_data = param.output->template mutable_data<float>();
  int index_size = param.indexs->dims()[0];
  auto update_dims = param.updates->dims();
  auto in_dims = param.x->dims();
  int num = 1;
  for (int i = 1; i < update_dims.size(); i++) {
    num *= update_dims[i];
  }
  std::vector<int64_t> input_offset;
  input_offset.resize(in_dims.size());
  input_offset[in_dims.size() - 1] = 1;
  for (int i = in_dims.size() - 2; i >= 0; i--) {
    input_offset[i] = input_offset[i + 1] * in_dims[i + 1];
  }
  host::memcpy(output_data, din_data, sizeof(float) * param.x->numel());
  scatter_nd_add(indexs_data,
                 updates_data,
                 output_data,
                 input_offset,
                 index_size,
                 num,
                 in_dims.size());
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(scatter_nd_add,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::ScatterNdAddCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();
