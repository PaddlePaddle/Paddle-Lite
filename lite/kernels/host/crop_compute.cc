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

#include "lite/kernels/host/crop_compute.h"
#include <vector>
#include "lite/backends/host/math/slice.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T, PrecisionType PType>
void CropCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::CropParam>();
  const lite::Tensor* x = param.X;
  lite::Tensor* out = param.Out;

  auto out_shape = out->dims().Vectorize();
  std::vector<int> shape = std::vector<int>(out_shape.begin(), out_shape.end());
  std::vector<int> starts;
  if (param.Offsets != nullptr) {
    auto offset_data = param.Offsets->template data<int>();
    for (int64_t i = 0; i < param.Offsets->numel(); i++) {
      starts.push_back(offset_data[i]);
    }
  } else {
    starts = param.offsets;
  }

  std::vector<int> ends;
  std::vector<int> axes;
  for (size_t i = 0; i < starts.size(); i++) {
    ends.push_back(starts[i] + shape[i]);
    axes.push_back(i);
  }

  lite::host::math::slice(x->template data<T>(),
                          x->dims().Vectorize(),
                          axes,
                          starts,
                          ends,
                          out->template mutable_data<T>());

  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using crop_float =
    paddle::lite::kernels::host::CropCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(crop, kHost, kFloat, kAny, crop_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Offsets",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using crop_int32 =
    paddle::lite::kernels::host::CropCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(crop, kHost, kInt32, kAny, crop_int32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Offsets",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();
