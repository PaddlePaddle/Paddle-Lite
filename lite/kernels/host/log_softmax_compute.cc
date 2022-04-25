// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/log_softmax_compute.h"
#include <cmath>
#include <limits>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void LogSoftmaxCompute::Run() {
  auto& param = this->Param<operators::LogSoftmaxParam>();

  auto x_dims_ = param.x->dims();
  int dim_size = param.x->dims().size();
  auto x_rank = x_dims_.size();
  auto axis = (param.axis < 0) ? param.axis + dim_size : param.axis;

  const auto* x_data = param.x->data<float>();
  auto* out_data = param.output->mutable_data<float>();

  if (axis < 0) {
    axis += x_rank;
  }
  int axis_size = x_dims_[axis];
  int outer_num = x_dims_.Slice(0, axis).production();
  int inner_num = x_dims_.Slice(axis + 1, x_rank).production();
  int compute_size = outer_num * inner_num;
  for (int i = 0; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int start = idx_outer * inner_num + idx_inner;
    int offset;

    offset = start;
    float max_data = std::numeric_limits<float>::lowest();
    for (int j = 0; j < axis_size; j++) {
      max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
      offset += inner_num;
    }

    offset = start;
    float sum_data = 0.f;
    for (int j = 0; j < axis_size; j++) {
      out_data[offset] = std::exp(x_data[offset] - max_data);
      sum_data += out_data[offset];
      offset += inner_num;
    }

    offset = start;
    for (int j = 0; j < axis_size; j++) {
      out_data[offset] /= sum_data;
      out_data[offset] = std::log(out_data[offset]);
      offset += inner_num;
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(log_softmax,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::LogSoftmaxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
