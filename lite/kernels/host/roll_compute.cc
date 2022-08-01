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

#include "lite/kernels/host/roll_compute.h"
#include <cmath>
#include <limits>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
inline void ShiftAlongDim(T* data,
                          const DDim& input_dim,
                          int64_t dim,
                          int64_t shift) {
  if (dim < 0) {
    dim += input_dim.size();
  }
  if (input_dim[dim] == 0) {
    return;
  }
  shift = shift % input_dim[dim];
  if (shift < 0) {
    shift += input_dim[dim];
  }

  auto outer_loops = 1;
  for (auto i = 0; i < dim; i++) {
    outer_loops *= input_dim[i];
  }
  auto slice_width = 1;
  for (auto i = dim + 1; i < input_dim.size(); i++) {
    slice_width *= input_dim[i];
  }

  if (shift == 0) {
    return;
  }

  std::vector<T> head;
  auto head_size = slice_width * (input_dim[dim] - shift);
  head.resize(head_size);

  for (auto i = 0; i < outer_loops; i++) {
    for (auto j = 0; j < head_size; j++) {
      head[j] = data[i * input_dim[dim] * slice_width + j];
    }
    for (auto j = input_dim[dim] - shift; j < input_dim[dim]; j++) {
      auto dst_pos = j - input_dim[dim] + shift;
      for (auto k = 0; k < slice_width; k++) {
        data[(i * input_dim[dim] + dst_pos) * slice_width + k] =
            data[(i * input_dim[dim] + j) * slice_width + k];
      }
    }
    for (auto j = 0; j < head_size; j++) {
      data[(i * input_dim[dim] + shift) * slice_width + j] = head[j];
    }
  }
}

void RollCompute::Run() {
  auto& param = this->Param<operators::RollParam>();

  const auto* x = param.X;
  auto* out = param.Out;
  std::vector<int64_t> axis = param.axis;
  std::vector<int64_t> shifts;
  if (param.ShiftsTensor != nullptr) {
    auto shift_data = param.ShiftsTensor->template data<int64_t>();
    for (int64_t i = 0; i < param.ShiftsTensor->numel(); i++) {
      shifts.push_back(shift_data[i]);
    }
  } else {
    shifts = param.shifts;
  }

  int nums = shifts.size();
  DDim input_dim = x->dims();
  int input_size = input_dim.size();
  // axis = none, reshape to 1-D tensor
  if (axis.size() == 0) {
    axis.push_back(0l);
    input_dim = DDim(std::vector<int64_t>({input_size}));
  }

  out->CopyDataFrom(*x);
  auto* out_data = param.Out->mutable_data<float>();
  for (size_t i = 0; i < nums; i++) {
    int64_t input_size = input_dim.size();
    CHECK_GE(axis[i], -input_size) << "axis[i]: " << axis[i]
                                   << ", input_dim.size(): " << input_size;
    CHECK_LT(axis[i], input_size) << "axis[i]: " << axis[i]
                                  << ", input_dim.size(): " << input_size;
    ShiftAlongDim(out_data, input_dim, axis[i], shifts[i]);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    roll, kHost, kFloat, kAny, paddle::lite::kernels::host::RollCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("ShiftsTensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();
