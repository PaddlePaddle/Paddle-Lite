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
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define COMPARE_FUNCTOR(name, op)                                           \
  template <typename T>                                                     \
  struct _##name##Functor {                                                 \
    inline bool operator()(const T &a, const T &b) const { return a op b; } \
  };

COMPARE_FUNCTOR(Equal, ==);
COMPARE_FUNCTOR(NotEqual, !=);
COMPARE_FUNCTOR(LessThan, <);
COMPARE_FUNCTOR(LessEqual, <=);
COMPARE_FUNCTOR(GreaterThan, >);
COMPARE_FUNCTOR(GreaterEqual, >=);

template <>
struct _EqualFunctor<float> {
  inline bool operator()(const float &a, const float &b) const {
    // It is safe to cast a and b to double.
    return fabs(static_cast<double>(a - b)) < 1e-8;
  }
};

template <>
struct _NotEqualFunctor<float> {
  inline bool operator()(const float &a, const float &b) const {
    return !_EqualFunctor<float>()(a, b);
  }
};

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
template <template <typename T> class Functor>
void CompareCompute<Functor>::PrepareForRun() {}

template <template <typename T> class Functor>
void CompareCompute<Functor>::Run() {
  auto &param = this->Param<operators::CompareParam>();

  using CompareFunctor = Functor<float>;

  const size_t x_size = param.X->numel();
  const size_t y_size = param.Y->numel();
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  bool *z = param.Out->template mutable_data<bool>();
  const auto *x = param.X->template data<int>();
  const auto *y = param.Y->template data<float>();
  auto axis = param.axis;
  bool force_cpu = param.force_cpu;
  if (x_size == y_size) {
    for (int i = 0; i < x_size; ++i) {
      z[i] = CompareFunctor()(x[i], y[i]);
      // z[i] = x[i] < y[i];
    }
  } else {
    int axis = (param.axis == -1 ? x_dims.size() - y_dims.size() : param.axis);
    int outer_num, mid_num, inner_num;
    get_mid_dims(x_dims, y_dims, axis, &outer_num, &mid_num, &inner_num);
    for (int outer_id = 0; outer_id < outer_num; ++outer_id) {
      for (int mid_id = 0; mid_id < mid_num; ++mid_id) {
        auto y_data = y[mid_id];
        for (int inner_id = 0; inner_id < inner_num; ++inner_id) {
          int index = (outer_id * mid_num + mid_id) * inner_num + inner_id;
          z[index] = CompareFunctor()(x[index], y_data);
          // z[index] = x[index] < y_data;
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
                     paddle::lite::kernels::arm::CompareCompute<
                         paddle::lite::kernels::arm::_LessThanFunctor>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
REGISTER_LITE_KERNEL(equal,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::CompareCompute<
                         paddle::lite::kernels::arm::_EqualFunctor>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
REGISTER_LITE_KERNEL(not_equal,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::CompareCompute<
                         paddle::lite::kernels::arm::_NotEqualFunctor>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
REGISTER_LITE_KERNEL(less_equal,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::CompareCompute<
                         paddle::lite::kernels::arm::_LessEqualFunctor>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
REGISTER_LITE_KERNEL(greater_than,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::CompareCompute<
                         paddle::lite::kernels::arm::_GreaterThanFunctor>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
REGISTER_LITE_KERNEL(greater_equal,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::CompareCompute<
                         paddle::lite::kernels::arm::_GreaterEqualFunctor>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
