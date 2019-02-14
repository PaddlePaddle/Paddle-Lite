/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "operators/kernel/tensor_array_read_write_kernel.h"

namespace paddle_mobile {
namespace operators {

#ifdef WRITE_TO_ARRAY_OP
template <>
bool WriteToArrayKernel<CPU, float>::Init(WriteToArrayParam<CPU> *param) {
  return true;
}

template <>
void WriteToArrayKernel<CPU, float>::Compute(
    const WriteToArrayParam<CPU> &param) {
  int64_t offset = param.index_->data<int64_t>()[0];
  if (offset >= param.output_->size()) {
    param.output_->resize(offset + 1);
  }

  framework::LoDTensor *out_tensor = &(param.output_->at(offset));
  out_tensor->set_lod(param.input_->lod());
  if (param.input_->memory_size() > 0) {
    TensorCopy(*(param.input_), out_tensor);
  }
}
#endif  // WRITE_TO_ARRAY_OP

#ifdef READ_FROM_ARRAY_OP
template <>
bool ReadFromArrayKernel<CPU, float>::Init(ReadFromArrayParam<CPU> *param) {
  return true;
}

template <>
void ReadFromArrayKernel<CPU, float>::Compute(
    const ReadFromArrayParam<CPU> &param) {
  int64_t offset = param.index_->data<int64_t>()[0];
  if (offset < param.input_->size()) {
    TensorCopy(param.input_->at(offset), param.output_);
    param.output_->set_lod(param.input_->at(offset).lod());
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(
        "Can not read tensor which index is `%d` since it only has `%d` inputs",
        offset, param.input_->size());
  }
}
#endif  // READ_FROM_ARRAY_OP

}  // namespace operators
}  // namespace paddle_mobile
