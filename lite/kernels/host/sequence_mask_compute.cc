// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/sequence_mask_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void SequenceMaskCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto* x = param.X;
  auto* y = parm.Y;
  int maxlen = param.maxlen;
  auto* max_len_tensor = param.MaxLenTensor;
  if (max_len_tensor != nullptr) {
    maxlen = max_len_tensor->template data<int>()[0];
    CHECK_GT(maxlen, 0) << "Input(MaxLenTensor)'s value should be greater than "
                           "0. But received maxlen: "
                        << maxlen;
  }

  auto* x_data = x->template data<T>();
  auto x_size = x->numel();
  if (maxlen < 0) {
    maxlen = static_cast<int>(*std::max_element(x_data, x_data + x_size));
  }

  auto y_shape = x->dims().Vectorize();
  y_shape.push_back(static_cast<int64_t>(maxlen));
  y->Resize(y_shape);
  y->set_lod(x->lod());

  int out_type = param.out_dtype;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_mask,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SequenceMaskCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("MaxLenTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(sequence_mask,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SequenceMaskCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("MaxLenTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(sequence_mask,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SequenceMaskCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("MaxLenTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
