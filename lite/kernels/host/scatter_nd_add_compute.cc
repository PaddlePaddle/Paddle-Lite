// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/scatter_nd_add_compute.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, typename IndexType>
void ScatterNdAdd(const IndexType* indexs,
                  const T* updates,
                  T* dst,
                  std::vector<int> x_dims_offset,
                  int index_size,
                  int index_count,
                  int add_size) {
  int index_offset = index_size / index_count;
  for (int i = 0; i < index_count; i++) {
    int dst_offset = 0;
    for (int j = 0; j < index_offset; j++) {
      dst_offset += indexs[j] * x_dims_offset[j];
    }
    indexs += index_offset;
    T* dst_tmp = dst + dst_offset;
    for (int j = 0; j < add_size; j++) {
      dst_tmp[j] += updates[j];
    }
    updates += add_size;
  }
}

template <typename T, typename IndexType>
void ScatterNdAddCompute<T, IndexType>::Run() {
  auto& param = this->template Param<param_t>();
  const T* din_data = param.x->template data<T>();
  const T* updates_data = param.updates->template data<T>();
  const IndexType* indexs_data = param.indexs->template data<IndexType>();
  T* output_data = param.output->template mutable_data<T>();
  memcpy(output_data, din_data, sizeof(T) * param.x->numel());

  auto x_dims = param.x->dims();
  auto index_dims = param.indexs->dims();
  auto update_dims = param.updates->dims();
  int index_size = static_cast<int>(index_dims.production());
  int index_count = index_dims.count(0, index_dims.size() - 1);
  int index_step = index_size / index_count;

  std::vector<int> x_dims_offset(x_dims.size());
  x_dims_offset[x_dims_offset.size() - 1] = 1;
  for (int i = static_cast<int>(x_dims.size()) - 2; i >= 0; i--) {
    x_dims_offset[i] = x_dims_offset[i + 1] * x_dims[i + 1];
  }

  int add_size = x_dims.count(index_step, x_dims.size());

  ScatterNdAdd(indexs_data,
               updates_data,
               output_data,
               x_dims_offset,
               index_size,
               index_count,
               add_size);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using ScatterNdAddFloat32Int32 =
    paddle::lite::kernels::host::ScatterNdAddCompute<float, int>;
REGISTER_LITE_KERNEL(
    scatter_nd_add, kHost, kFloat, kNCHW, ScatterNdAddFloat32Int32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

using ScatterNdAddFloat32Int64 =
    paddle::lite::kernels::host::ScatterNdAddCompute<float, int64_t>;
REGISTER_LITE_KERNEL(scatter_nd_add,
                     kHost,
                     kFloat,
                     kNCHW,
                     ScatterNdAddFloat32Int64,
                     float32_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

using ScatterNdAddInt32Int32 =
    paddle::lite::kernels::host::ScatterNdAddCompute<int, int>;
REGISTER_LITE_KERNEL(
    scatter_nd_add, kHost, kFloat, kNCHW, ScatterNdAddInt32Int32, int32_int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using ScatterNdAddInt32Int64 =
    paddle::lite::kernels::host::ScatterNdAddCompute<int, int64_t>;
REGISTER_LITE_KERNEL(
    scatter_nd_add, kHost, kFloat, kNCHW, ScatterNdAddInt32Int64, int32_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using ScatterNdAddInt64Int32 =
    paddle::lite::kernels::host::ScatterNdAddCompute<int64_t, int>;
REGISTER_LITE_KERNEL(
    scatter_nd_add, kHost, kFloat, kNCHW, ScatterNdAddInt64Int32, int64_int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

using ScatterNdAddInt64Int64 =
    paddle::lite::kernels::host::ScatterNdAddCompute<int64_t, int64_t>;
REGISTER_LITE_KERNEL(
    scatter_nd_add, kHost, kFloat, kNCHW, ScatterNdAddInt64Int64, int64_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
