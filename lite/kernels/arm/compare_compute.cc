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

#include "lite/kernels/arm/compare_compute.h"
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

inline void get_mid_dims(const lite::DDim &x_dims,
                         const lite::DDim &y_dims,
                         const int axis,
                         int *pre,
                         int *n,
                         int *post) {
  *pre = 1;
  *n = 1;
  *post = 1;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }

  for (int i = 0; i < y_dims.size(); ++i) {
    (*n) *= y_dims[i];
  }

  for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
}

void LessThanCompute::PrepareForRun() {}

void LessThanCompute::Run() {
  auto &param = this->Param<operators::CompareParam>();

  ///  using LogicalFunctor = Functor<bool>;

  const size_t x_size = param.X->numel();
  const size_t y_size = param.Y->numel();
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  bool *z = param.Out->mutable_data<bool>();
  const float *x = param.X->data<float>();
  const float *y = param.Y->data<float>();
  auto axis = param.axis;
  bool force_cpu = param.force_cpu;
  if (x_size == y_size) {
    for (int i = 0; i < x_size; ++i) {
      // z[i] = CompareFunctor()(x[i], y[i]);
      z[i] = x[i] < y[i];
    }
  } else {
    int axis = (param.axis == -1 ? x_dims.size() - y_dims.size() : param.axis);
    int outer_num, mid_num, inner_num;
    get_mid_dims(x_dims, y_dims, axis, &outer_num, &mid_num, &inner_num);
    for (int outer_id = 0; outer_id < outer_num; ++outer_id) {
      for (int mid_id = 0; mid_id < mid_num; ++mid_id) {
        float y_data = y[mid_id];
        for (int inner_id = 0; inner_id < inner_num; ++inner_id) {
          int index = (outer_id * mid_num + mid_id) * inner_num + inner_id;
          // z[index] = CompareFunctor()(x[index], y_data);
          z[index] = x[index] < y_data;
        }
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(less_than,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::LessThanCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
