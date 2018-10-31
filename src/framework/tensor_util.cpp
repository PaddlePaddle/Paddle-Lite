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

#include "tensor_util.h"

namespace paddle_mobile {
namespace framework {

void TensorCopy(const Tensor &src, Tensor *dst) {
  src.check_memory_size();
  dst->Resize(src.dims());
  auto src_ptr = src.data<void>();
  auto dst_ptr = dst->mutable_data(src.type());
  auto size = src.numel() * SizeOfType(src.type());
  memory::Copy(dst_ptr, src_ptr, size);
}

}  // namespace framework
}  // namespace paddle_mobile
