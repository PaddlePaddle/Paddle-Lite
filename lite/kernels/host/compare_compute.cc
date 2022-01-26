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
#include <algorithm>
#include <cstdlib>
#include <functional>
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

inline int GetElementwiseIndex(const int64_t *x_dims_array,
                               const int max_dim,
                               const int *index_array) {
  int index_ = 0;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] > 1) {
      index_ = index_ * x_dims_array[i] + index_array[i];
    }
  }
  return index_;
}

inline void UpdateElementwiseIndexArray(const int64_t *out_dims_array,
                                        const int max_dim,
                                        int *index_array) {
  for (int i = max_dim - 1; i >= 0; --i) {
    ++index_array[i];
    if (index_array[i] >= out_dims_array[i]) {
      index_array[i] -= out_dims_array[i];
    } else {
      break;
    }
  }
}

template <typename Functor, typename T, typename OutType = T>
void CommonForwardBroadcast(const T *x_data,
                            const T *y_data,
                            OutType *out_data,
                            int64_t *x_dims_array,
                            int64_t *y_dims_array,
                            int64_t *out_dims_array,
                            int max_dim,
                            Functor func) {
  std::vector<int> index_array(max_dim, 0);

  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int64_t>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = GetElementwiseIndex(x_dims_array, max_dim, index_array.data());
    y_index = GetElementwiseIndex(y_dims_array, max_dim, index_array.data());
    out_data[out_index] = func(x_data[x_index], y_data[y_index]);

    UpdateElementwiseIndexArray(out_dims_array, max_dim, index_array.data());
  }
}

inline lite::DDim trim_trailing_singular_dims(const lite::DDim &dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }
  if (actual_dims_size == dims.size()) return dims;
  std::vector<int64_t> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (int i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return lite::DDim();
  }
  lite::DDim actual_dims = lite::DDim(trim_dims);
  return actual_dims;
}

template <typename Functor, typename T, typename OutType = T>
void CommonElementwiseBroadcastForward(const T *x,
                                       const T *y,
                                       OutType *z,
                                       const DDim &x_dims,
                                       const DDim &y_dims,
                                       const DDim &out_dims,
                                       Functor func) {
  int max_dim = std::max(x_dims.size(), y_dims.size());

  std::vector<int64_t> x_dims_array = x_dims.Vectorize();
  std::vector<int64_t> y_dims_array = y_dims.Vectorize();
  std::vector<int64_t> out_dims_array = out_dims.Vectorize();

  CommonForwardBroadcast<Functor, T, OutType>(x,
                                              y,
                                              z,
                                              x_dims_array.data(),
                                              y_dims_array.data(),
                                              out_dims_array.data(),
                                              max_dim,
                                              func);
}

inline void get_mid_dims(const lite::DDim &x_dims,
                         const lite::DDim &y_dims,
                         const int axis,
                         int *pre,
                         int *n,
                         int *post,
                         int *is_run_common_broadcast) {
  *pre = 1;
  *n = 1;
  *post = 1;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }

  for (int i = 0; i < y_dims.size(); ++i) {
    // do broadcast
    if (x_dims[i + axis] != y_dims[i]) {
      *is_run_common_broadcast = 1;
    }
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
    int axis = (param.axis == -1 ? abs(static_cast<int>(x_dims.size()) -
                                       static_cast<int>(y_dims.size()))
                                 : param.axis);
    // If Y contains only one data, all_broad_cast mode will be applied.
    // In this mode, each member in X will compare to the only var in Y.
    if (param.Y->numel() == 1) {
      axis = x_dims.size();
    }
    int outer_num, mid_num, inner_num;
    int is_run_common_broadcast = 0;
    int axis_trim = 0;
    auto y_dims_trimed = trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    get_mid_dims(x_dims,
                 y_dims_trimed,
                 axis_trim,
                 &outer_num,
                 &mid_num,
                 &inner_num,
                 &is_run_common_broadcast);
    if (is_run_common_broadcast == 1) {
      CommonElementwiseBroadcastForward<CompareFunctor, DType, bool>(
          x, y, z, x_dims, y_dims, param.Out->dims(), CompareFunctor());
      return;
    }

    // get_mid_dims(x_dims, y_dims, axis, &outer_num, &mid_num, &inner_num);
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
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("equal", 1)
    .Finalize();

using equal_int64 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kInt64),
    paddle::lite::kernels::host::_EqualFunctor<int64_t>>;
REGISTER_LITE_KERNEL(equal, kHost, kInt64, kAny, equal_int64, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("equal", 1)
    .Finalize();

// float kernel has higher score when picking kernel.
// TODO(zhupengyang): merge equal_int64 later
using equal_int64_f = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_EqualFunctor<int64_t>>;
REGISTER_LITE_KERNEL(equal, kHost, kFloat, kAny, equal_int64_f, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("equal", 1)
    .Finalize();

using equal_int32 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kInt32),
    paddle::lite::kernels::host::_EqualFunctor<int32_t>>;
REGISTER_LITE_KERNEL(equal, kHost, kInt32, kAny, equal_int32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("equal", 1)
    .Finalize();

// float kernel has higher score when picking kernel.
using equal_int32_f = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_EqualFunctor<int32_t>>;
REGISTER_LITE_KERNEL(equal, kHost, kFloat, kAny, equal_int32_f, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("equal", 1)
    .Finalize();

using not_equal_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_NotEqualFunctor<float>>;
REGISTER_LITE_KERNEL(not_equal, kHost, kFloat, kAny, not_equal_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("not_equal", 1)
    .Finalize();

using not_equal_int32 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_NotEqualFunctor<int32_t>>;
REGISTER_LITE_KERNEL(not_equal, kHost, kFloat, kAny, not_equal_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("not_equal", 1)
    .Finalize();

using not_equal_int64 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_NotEqualFunctor<int64_t>>;
REGISTER_LITE_KERNEL(not_equal, kHost, kFloat, kAny, not_equal_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("not_equal", 1)
    .Finalize();

using less_than_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_LessThanFunctor<float>>;
REGISTER_LITE_KERNEL(less_than, kHost, kFloat, kAny, less_than_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

using less_than_int32 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kInt32),
    paddle::lite::kernels::host::_LessThanFunctor<int32_t>>;
REGISTER_LITE_KERNEL(less_than, kHost, kInt32, kAny, less_than_int32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

// float kernel has higher score when picking kernel.
using less_than_int32_f = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_LessThanFunctor<int32_t>>;
REGISTER_LITE_KERNEL(less_than, kHost, kFloat, kAny, less_than_int32_f, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

using less_than_int64 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kInt64),
    paddle::lite::kernels::host::_LessThanFunctor<int64_t>>;
REGISTER_LITE_KERNEL(less_than, kHost, kInt64, kAny, less_than_int64, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

// float kernel has higher score when picking kernel.
using less_than_int64_f = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_LessThanFunctor<int64_t>>;
REGISTER_LITE_KERNEL(less_than, kHost, kFloat, kAny, less_than_int64_f, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

using less_equal_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_LessEqualFunctor<float>>;
REGISTER_LITE_KERNEL(less_equal, kHost, kFloat, kAny, less_equal_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_equal", 1)
    .Finalize();

using less_equal_int64 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kInt64),
    paddle::lite::kernels::host::_LessEqualFunctor<int64_t>>;
REGISTER_LITE_KERNEL(less_equal, kHost, kInt64, kAny, less_equal_int64, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_equal", 1)
    .Finalize();

// float kernel has higher priority
using less_equal_int64_f = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_LessEqualFunctor<int64_t>>;
REGISTER_LITE_KERNEL(less_equal, kHost, kFloat, kAny, less_equal_int64_f, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_equal", 1)
    .Finalize();

// float kernel has higher priority
using less_equal_int32_f = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_LessEqualFunctor<int32_t>>;
REGISTER_LITE_KERNEL(less_equal, kHost, kFloat, kAny, less_equal_int32_f, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_equal", 1)
    .Finalize();

using greater_than_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_GreaterThanFunctor<float>>;
REGISTER_LITE_KERNEL(greater_than, kHost, kFloat, kAny, greater_than_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_than", 1)
    .Finalize();

using greater_than_bool = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_GreaterThanFunctor<bool>>;
REGISTER_LITE_KERNEL(
    greater_than, kHost, kFloat, kAny, greater_than_bool, def_bool)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_than", 1)
    .Finalize();

using greater_than_int32 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_GreaterThanFunctor<int32_t>>;
REGISTER_LITE_KERNEL(
    greater_than, kHost, kFloat, kAny, greater_than_int32, def_int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_than", 1)
    .Finalize();

using greater_than_int64 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kInt64),
    paddle::lite::kernels::host::_GreaterThanFunctor<int64_t>>;
REGISTER_LITE_KERNEL(greater_than, kHost, kInt64, kAny, greater_than_int64, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_than", 1)
    .Finalize();

// float kernel has higher priority
using greater_than_int64_f = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_GreaterThanFunctor<int64_t>>;
REGISTER_LITE_KERNEL(
    greater_than, kHost, kFloat, kAny, greater_than_int64_f, def_int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_than", 1)
    .Finalize();

using greater_equal_float = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_GreaterEqualFunctor<float>>;
REGISTER_LITE_KERNEL(
    greater_equal, kHost, kFloat, kAny, greater_equal_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_equal", 1)
    .Finalize();

using greater_equal_int64 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_GreaterEqualFunctor<int64_t>>;
REGISTER_LITE_KERNEL(
    greater_equal, kHost, kFloat, kAny, greater_equal_int64, def_int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_equal", 1)
    .Finalize();

using greater_equal_int32 = paddle::lite::kernels::host::CompareCompute<
    PRECISION(kFloat),
    paddle::lite::kernels::host::_GreaterEqualFunctor<int32_t>>;
REGISTER_LITE_KERNEL(
    greater_equal, kHost, kFloat, kAny, greater_equal_int32, def_int32)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .BindPaddleOpVersion("greater_equal", 1)
    .Finalize();
