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

#include "lite/kernels/host/expand_v2_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, PrecisionType PType>
void ExpandV2Compute<T, PType>::Run() {
  auto& param = this->template Param<operators::ExpandV2Param>();
  const auto* x = param.X;
  auto* out = param.Out;
  std::vector<int> expand_shape;
  if (param.Shape != nullptr) {
    auto Shape_data = param.Shape->template data<int>();
    for (int64_t i = 0; i < param.Shape->numel(); i++) {
      expand_shape.push_back(Shape_data[i]);
    }
  } else if (!param.expand_shapes_tensor.empty()) {
    for (size_t i = 0; i < param.expand_shapes_tensor.size(); i++) {
      expand_shape.push_back(
          param.expand_shapes_tensor[i]->template data<int>()[0]);
    }
  } else {
    expand_shape = param.shape;
  }
  std::vector<int64_t> vec_in_dims;
  DDim in_shape = x->dims();
  for (int i = 0; i < in_shape.size(); ++i) {
    vec_in_dims.push_back(static_cast<int64_t>(in_shape[i]));
  }
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int> repeat_times(vec_in_dims.size());
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    if (i < diff) {
      repeat_times[i] = expand_shape[i];
    } else if (expand_shape[i] > 0) {
      if (vec_in_dims[i] != 1) {
        repeat_times[i] = 1;
      } else {
        repeat_times[i] = expand_shape[i];
      }
    } else {
      repeat_times[i] = 1;
    }
  }
  const T* src = x->template data<T>();
  T* dst = out->template mutable_data<T>();
  DDim new_in_shape;
  new_in_shape.ConstructFrom(vec_in_dims);
  int dims = repeat_times.size();
  DDim out_shape = out->dims();
  int inner_num = 1;
  int index = dims - 1;
  int outer_num = new_in_shape.count(0, index);
  inner_num *= new_in_shape[index];
  for (int j = 0; j < outer_num; ++j) {
    for (int k = 0; k < repeat_times[index]; ++k) {
      memcpy(dst + (j * repeat_times[index] + k) * inner_num,
             src + j * inner_num,
             sizeof(T) * inner_num);
    }
  }
  inner_num *= repeat_times[index];
  for (int index = dims - 2; index >= 0; --index) {
    int outer_num = new_in_shape.count(0, index);
    inner_num *= new_in_shape[index];
    for (int j = outer_num - 1; j >= 0; --j) {
      for (int k = repeat_times[index] - 1; k >= 0; --k) {
        memcpy(dst + (j * repeat_times[index] + k) * inner_num,
               dst + j * inner_num,
               sizeof(T) * inner_num);
      }
    }
    inner_num *= repeat_times[index];
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using expand_v2_float =
    paddle::lite::kernels::host::ExpandV2Compute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(expand_v2, kHost, kFloat, kAny, expand_v2_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_shapes_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using expand_v2_int32 =
    paddle::lite::kernels::host::ExpandV2Compute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(expand_v2, kHost, kFloat, kAny, expand_v2_int32, def_int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_shapes_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using expand_v2_int64 =
    paddle::lite::kernels::host::ExpandV2Compute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(expand_v2, kHost, kFloat, kAny, expand_v2_int64, def_int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_shapes_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
