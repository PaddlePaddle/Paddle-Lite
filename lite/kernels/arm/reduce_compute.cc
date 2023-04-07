// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/reduce_compute.h"
#include <string>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/reduce.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename T, paddle::lite::arm::math::ReduceProcessType OpType>
void ReduceCompute<T, OpType>::Run() {
  auto& param = Param<operators::ReduceParam>();
  const T* input = param.X->template data<T>();
  auto x_dims = param.X->dims();
  int x_rank = x_dims.size();
  T* output = param.Out->template mutable_data<T>();
  auto out_dims = param.Out->dims();
  auto dim = param.dim;
  bool reduce_all = param.reduce_all;
  bool keep_dim = param.keep_dim;
  auto vec_xdims = x_dims.Vectorize();
  auto vec_odims = out_dims.Vectorize();
  CHECK(x_rank <= 6) << "Only support input_dim <= 6 for now.";
  param.Out->set_precision(param.X->precision());
  if (!dim.empty()) {
    for (int i = 0; i < dim.size(); i++) {
      if (dim[i] < 0) {
        dim[i] += x_rank;
      }
    }
  }
  lite::arm::math::ReduceImpl<T>(
      input, vec_xdims, output, vec_odims, dim, reduce_all, OpType);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using int64_reduce_mean = paddle::lite::kernels::arm::
    ReduceCompute<int64_t, paddle::lite::arm::math::ReduceProcessType::mean>;
REGISTER_LITE_KERNEL(reduce_mean, kARM, kFloat, kNCHW, int64_reduce_mean, i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

using int32_reduce_mean = paddle::lite::kernels::arm::
    ReduceCompute<int, paddle::lite::arm::math::ReduceProcessType::mean>;
REGISTER_LITE_KERNEL(reduce_mean, kARM, kFloat, kNCHW, int32_reduce_mean, i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using float_reduce_mean = paddle::lite::kernels::arm::
    ReduceCompute<float, paddle::lite::arm::math::ReduceProcessType::mean>;
REGISTER_LITE_KERNEL(reduce_mean, kARM, kFloat, kNCHW, float_reduce_mean, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

#ifdef LITE_BUILD_EXTRA
using float_reduce_max = paddle::lite::kernels::arm::
    ReduceCompute<float, paddle::lite::arm::math::ReduceProcessType::max>;
REGISTER_LITE_KERNEL(reduce_max, kARM, kFloat, kNCHW, float_reduce_max, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using int64_reduce_max = paddle::lite::kernels::arm::
    ReduceCompute<int64_t, paddle::lite::arm::math::ReduceProcessType::max>;
REGISTER_LITE_KERNEL(reduce_max, kARM, kFloat, kNCHW, int64_reduce_max, i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

using int32_reduce_max = paddle::lite::kernels::arm::
    ReduceCompute<int, paddle::lite::arm::math::ReduceProcessType::max>;
REGISTER_LITE_KERNEL(reduce_max, kARM, kFloat, kNCHW, int32_reduce_max, i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using float_reduce_min = paddle::lite::kernels::arm::
    ReduceCompute<float, paddle::lite::arm::math::ReduceProcessType::min>;
REGISTER_LITE_KERNEL(reduce_min, kARM, kFloat, kNCHW, float_reduce_min, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using int64_reduce_min = paddle::lite::kernels::arm::
    ReduceCompute<int64_t, paddle::lite::arm::math::ReduceProcessType::min>;
REGISTER_LITE_KERNEL(reduce_min, kARM, kFloat, kNCHW, int64_reduce_min, i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

using int32_reduce_min = paddle::lite::kernels::arm::
    ReduceCompute<int, paddle::lite::arm::math::ReduceProcessType::min>;
REGISTER_LITE_KERNEL(reduce_min, kARM, kFloat, kNCHW, int32_reduce_min, i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using float_reduce_sum = paddle::lite::kernels::arm::
    ReduceCompute<float, paddle::lite::arm::math::ReduceProcessType::sum>;
REGISTER_LITE_KERNEL(reduce_sum, kARM, kFloat, kNCHW, float_reduce_sum, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using int64_reduce_sum = paddle::lite::kernels::arm::
    ReduceCompute<int64_t, paddle::lite::arm::math::ReduceProcessType::sum>;
REGISTER_LITE_KERNEL(reduce_sum, kARM, kFloat, kNCHW, int64_reduce_sum, i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

using int32_reduce_sum = paddle::lite::kernels::arm::
    ReduceCompute<int, paddle::lite::arm::math::ReduceProcessType::sum>;
REGISTER_LITE_KERNEL(reduce_sum, kARM, kFloat, kNCHW, int32_reduce_sum, i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using float_reduce_prod = paddle::lite::kernels::arm::
    ReduceCompute<float, paddle::lite::arm::math::ReduceProcessType::prod>;
REGISTER_LITE_KERNEL(reduce_prod, kARM, kFloat, kNCHW, float_reduce_prod, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using int64_reduce_prod = paddle::lite::kernels::arm::
    ReduceCompute<int64_t, paddle::lite::arm::math::ReduceProcessType::prod>;
REGISTER_LITE_KERNEL(reduce_prod, kARM, kFloat, kNCHW, int64_reduce_prod, i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

using int32_reduce_prod = paddle::lite::kernels::arm::
    ReduceCompute<int, paddle::lite::arm::math::ReduceProcessType::prod>;
REGISTER_LITE_KERNEL(reduce_prod, kARM, kFloat, kNCHW, int32_reduce_prod, i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA
