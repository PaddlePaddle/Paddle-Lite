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

#include "lite/kernels/host/split_compute.h"
#include <vector>
#include "lite/backends/host/math/split.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, PrecisionType PType>
void SplitCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::SplitParam>();
  const T* din = param.x->template data<T>();
  auto& dout = param.output;
  auto in_dim = param.x->dims();

  std::vector<int> in_strides(in_dim.size());
  in_strides[in_dim.size() - 1] = in_dim[in_dim.size() - 1];
  for (int i = in_dim.size() - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * in_dim[i];
  }
  int axis = param.axis;
  if (axis < 0) {
    axis += static_cast<int>(param.x->dims().size());
  }

  lite::host::math::split(din, dout, axis, in_strides);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
REGISTER_LITE_KERNEL(split, kHost, kFP16, kNCHW, SplitFP16T, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFP16))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SectionsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFP16))})
    .Finalize();

#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(split, kHost, kFloat, kNCHW, SplitFloat, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SectionsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(split, kHost, kFloat, kNCHW, SplitInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SectionsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(split, kHost, kFloat, kNCHW, SplitInt64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SectionsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(split, kHost, kInt64, kNCHW, SplitInt64T, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SectionsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
