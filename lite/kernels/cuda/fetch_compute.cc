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

#include "lite/kernels/cuda/fetch_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType Ptype>
void FetchCompute<T, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  auto* fetch_list = param.fetch_list;
  if (fetch_list->size() <= static_cast<size_t>(param.col)) {
    fetch_list->resize(param.col + 1);
  }

  int num = static_cast<int>(param.input->numel());
  auto& dst = fetch_list->at(param.col);
  dst.Resize(param.input->dims());
  auto output = dst.template mutable_data<T>();
  TargetW::MemcpyAsync(output,
                       param.input->template data<T>(),
                       num * sizeof(T),
                       IoDirection::DtoH,
                       stream);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::cuda::FetchCompute<float, PRECISION(kFloat)>
    FetchFp32;

// When the model ends with a cpu kernel, adding cuda's fetch kernel will add
// useless io_copy, so we just remove register operator.
