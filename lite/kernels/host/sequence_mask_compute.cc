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

template <class Tx, class Ty>
void SequenceMask(const Tx* x, Ty* y, const int x_size, const int max_len) {
  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < max_len; j++) {
      y[j] = static_cast<Ty>(static_cast<Tx>(j) < x[i] ? 1 : 0);
    }
    y += max_len;
  }
}

template <class T>
void SequenceMaskCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto* x = param.X;
  auto* y = param.Y;
  int max_len = param.maxlen;
  auto* max_len_tensor = param.MaxLenTensor;
  if (max_len_tensor != nullptr) {
    max_len = max_len_tensor->template data<int>()[0];
    CHECK_GT(max_len, 0)
        << "Input(MaxLenTensor)'s value should be greater than "
           "0. But received maxlen: "
        << max_len;
  }

  auto* x_data = x->template data<T>();
  int x_size = static_cast<int>(x->numel());
  if (max_len < 0) {
    max_len = static_cast<int>(*std::max_element(x_data, x_data + x_size));
  }

  auto y_shape = x->dims().Vectorize();
  y_shape.push_back(static_cast<int64_t>(max_len));
  y->Resize(y_shape);
  y->set_lod(x->lod());

  int out_type = param.out_dtype;
  switch (lite::core::FluidType(out_type)) {
    case lite::core::FluidType::FP32: {
      SequenceMask(x_data, y->template mutable_data<float>(), x_size, max_len);
      break;
    }
    case lite::core::FluidType::INT32: {
      SequenceMask(x_data, y->template mutable_data<int>(), x_size, max_len);
      break;
    }
    case lite::core::FluidType::INT64: {
      SequenceMask(
          x_data, y->template mutable_data<int64_t>(), x_size, max_len);
      break;
    }
    default:
      LOG(FATAL) << "unsupported out data type: " << out_type;
      break;
  }
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
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
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
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
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
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
