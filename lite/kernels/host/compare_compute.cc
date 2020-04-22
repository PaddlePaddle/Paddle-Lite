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

#include "lite/kernels/host/compare_compute.h"
#include <math.h>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

#define COMPARE_FUNCTOR(name, op)                                           \
  template <typename T>                                                     \
  struct _##name##Functor {                                                 \
    using TYPE = T;                                                         \
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
  using TYPE = float;
  inline bool operator()(const float &a, const float &b) const {
    // It is safe to cast a and b to double.
    return fabs(static_cast<double>(a - b)) < 1e-8;
  }
};

template <>
struct _NotEqualFunctor<float> {
  using TYPE = float;
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

template <PrecisionType PType, typename CompareFunctor>
void CompareCompute<PType, CompareFunctor>::Run() {
  auto &param = this->template Param<operators::CompareParam>();
  using DType = typename CompareFunctor::TYPE;
  const size_t x_size = param.X->numel();
  const size_t y_size = param.Y->numel();
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  bool *z = param.Out->template mutable_data<bool>();
  const auto *x = param.X->template data<DType>();
  const auto *y = param.Y->template data<DType>();
  if (x_size == y_size) {
    for (int i = 0; i < x_size; ++i) {
      z[i] = CompareFunctor()(x[i], y[i]);
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
        }
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using equal_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_EqualFunctor<float>>;
REGISTER_LITE_KERNEL(equal, kHost, kFloat, kAny, equal_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();

using equal_int32 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kInt32),
    paddle::lite::kernels::host::_EqualFunctor<int32_t>>;
REGISTER_LITE_KERNEL(equal, kHost, kInt32, kAny, equal_int32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();

using not_equal_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_NotEqualFunctor<float>>;
REGISTER_LITE_KERNEL(not_equal, kHost, kFloat, kAny, not_equal_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();

using less_than_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_LessThanFunctor<float>>;
REGISTER_LITE_KERNEL(less_than, kHost, kFloat, kAny, less_than_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();

using less_than_int32 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kInt32),
    paddle::lite::kernels::host::_LessThanFunctor<int32_t>>;
REGISTER_LITE_KERNEL(less_than, kHost, kInt32, kAny, less_than_int32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();

using less_than_int64 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kInt64),
    paddle::lite::kernels::host::_LessThanFunctor<int64_t>>;
REGISTER_LITE_KERNEL(less_than, kHost, kInt64, kAny, less_than_int64, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();

using less_equal_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_LessEqualFunctor<float>>;
REGISTER_LITE_KERNEL(less_equal, kHost, kFloat, kAny, less_equal_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();

using greater_than_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_GreaterThanFunctor<float>>;
REGISTER_LITE_KERNEL(greater_than, kHost, kFloat, kAny, greater_than_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();

using greater_equal_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_GreaterEqualFunctor<float>>;
REGISTER_LITE_KERNEL(
    greater_equal, kHost, kFloat, kAny, greater_equal_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .Finalize();
