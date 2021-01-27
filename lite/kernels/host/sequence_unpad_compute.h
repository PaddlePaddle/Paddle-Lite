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
#pragma once

#include <vector>
#include "lite/backends/host/math/sequence_padding.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

namespace math = paddle::lite::host::math;

template <typename T>
class SequenceUnpadCompute
    : public KernelLite<TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny)> {
 public:
  using param_t = operators::SequenceUnpadParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    auto& ctx = this->ctx_->template As<HostContext>();

    auto x_dims = param.X->dims();
    auto len_dims = param.Length->dims();

    auto* seq_len_ptr = param.Length->template data<int64_t>();
    int64_t batch_size = len_dims[0];
    std::vector<uint64_t> out_lod0(batch_size + 1, 0);
    for (int64_t i = 0; i < batch_size; ++i) {
      out_lod0[i + 1] = out_lod0[i] + seq_len_ptr[i];
    }
    paddle::lite::LoD out_lod;
    out_lod.push_back(out_lod0);

    int64_t out_dim0 = out_lod0.back();
    std::vector<int64_t> out_dims{out_dim0};
    if (x_dims.size() == 2) {
      out_dims.push_back(1);
    } else {
      for (size_t i = 2; i < x_dims.size(); ++i) {
        out_dims.push_back(x_dims[i]);
      }
    }
    param.Out->Resize(out_dims);
    param.Out->set_lod(out_lod);

    param.Out->template mutable_data<T>();
    int64_t padded_length = param.X->dims()[1];
    math::UnpaddingLoDTensorFunctor<lite::TargetType::kHost, T>()(
        ctx,
        *param.X,
        param.Out,
        padded_length,
        0,
        false,
        math::kBatchLengthWidth);
  }

  virtual ~SequenceUnpadCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
