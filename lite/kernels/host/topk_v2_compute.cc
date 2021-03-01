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

#include "lite/kernels/host/topk_v2_compute.h"
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
bool comp_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
}

void TopkV2Compute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
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
  int outer_size = x_dims.Count(0, axis);
  int axis_size = x_dims[axis];
  int inner_size = x_dims.Count(axis + 1, dim_size);
  int sum_size = axis_size * inner_size;
  LOG(INFO) << "axis: " << param.axis << ", k: " << k;
  LOG(INFO) << "inner_size: " << inner_size << ", axis_size: " << axis_size
            << ", outer_size: " << outer_size;
  for (int i = 0; i < dim_size; i++) {
    LOG(INFO) << x_dims[i];
  }
  for (int n = 0; n < outer_size; n++) {
    const float* in_data = x_data + n * sum_size;
    float* out_data = out_val + n * sum_size;
    int64_t* out_ind_data = out_ind + n * sum_size;
    for (int i = 0; i < inner_size; i++) {
      std::vector<std::pair<float, int>> vec;
      for (int j = 0; j < axis_size; j++) {
        vec.push_back(std::make_pair(in_data[j * outer_size + i], j));
      }
      std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp_func);
      for (int j = 0; j < k; j++) {
        out_data[j * outer_size + i] = vec[j].frist;
        out_ind_data[j * outer_size + i] = vec[j].second;
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    top_k, kHost, kFloat, kNCHW, paddle::lite::kernels::arm::TopkCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("K", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
