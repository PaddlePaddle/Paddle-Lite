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

#include "lite/kernels/host/topk_v2_compute.h"
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {
bool comp_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
}

void TopkV2Compute::Run() {
  auto& param = Param<operators::TopkParam>();
  const float* x_data = param.X->data<float>();
  float* out_val = param.Out->mutable_data<float>();
  auto out_ind = param.Indices->mutable_data<int64_t>();
  DDim x_dims = param.X->dims();
  int axis = param.axis;
  int dim_size = x_dims.size();
  int k = param.K;
  if (axis < 0) {
    axis += dim_size;
  }
  if (param.k_is_tensor) {
    k = param.KTensor->data<int>()[0];
  }
  int outer_size = x_dims.count(0, axis);
  int axis_size = x_dims[axis];
  int inner_size = x_dims.count(axis + 1, dim_size);
  int sum_size = axis_size * inner_size;
  int out_sum_size = k * inner_size;

  for (int i = 0; i < outer_size; i++) {
    for (int tmp_j = 0; tmp_j < inner_size; tmp_j++) {
      // we need sort outer_size * inner_size times
      // and every times we need sort `axis_size` float

      // we should start from here and pick
      // `axis_size` float strided by inner_size
      int glb_in_off = i * sum_size + tmp_j;
      std::vector<std::pair<float, int>> vec;
      for (int j = 0; j < axis_size; j++) {
        vec.push_back(std::make_pair(x_data[glb_in_off + j * inner_size], j));
      }
      std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp_func);

      // we should start from here and put
      // `k` float from here  strided by inner_size
      int glb_out_off = i * out_sum_size + tmp_j;

      for (int j = 0; j < k; j++) {
        out_val[glb_out_off + j * inner_size] = vec[j].first;
        out_ind[glb_out_off + j * inner_size] = vec[j].second;
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(top_k_v2,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::TopkV2Compute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("K", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
