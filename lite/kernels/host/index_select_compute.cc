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

#include "lite/kernels/host/index_select_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void Index_selectCompute<T>::Run() {
  auto& param = Param<operators::Index_selectParam>();
  lite::Tensor* input = param.X;
  lite::Tensor* index = param.Index;
  lite::Tensor* output = param.Out;

  auto input_ddim = input->dims();
  auto index_ddim = index->dims();
  auto output_ddim = output->dims();

  int left = input_ddim.count(0, param.dim);
  int middle = input_ddim[param.dim];
  int right = input_ddim.count(param.dim + 1, input_ddim.size());

  const T* in_ptr = input->data<T>();
  const int64_t* index_ptr = index->data<int64_t>();
  T* out_ptr = output->mutable_data<T>();

  for (int i = 0; i < left; i++)
    for (int k = 0; k < index_ddim.production(); k++)
      for (int j = 0; j < right; j++)
        out_ptr[i * index_ddim.production() * right + k * right + j] =
            in_ptr[i * middle * right + index_ptr[k] * right + j];

  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(index_select,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::Index_selectCompute<float>,
                     fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindPaddleOpVersion("index_select", 1)
    .Finalize();

#ifdef LITE_BUILD_EXTRA

REGISTER_LITE_KERNEL(index_select,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::Index_selectCompute<int32_t>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindPaddleOpVersion("index_select", 1)
    .Finalize();

REGISTER_LITE_KERNEL(index_select,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::Index_selectCompute<int16_t>,
                     int16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindPaddleOpVersion("index_select", 1)
    .Finalize();

REGISTER_LITE_KERNEL(index_select,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::Index_selectCompute<int8_t>,
                     int8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindPaddleOpVersion("index_select", 1)
    .Finalize();

#endif  // LITE_BUILD_EXTRA
