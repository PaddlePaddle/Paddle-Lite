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

#include <Eigen/Core>
#include <random>
#include <string>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
#include "lite/fluid/eigen.h"
#include "lite/operators/attention_padding_mask_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class AttentionPaddingMaskCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::AttentionPaddingMaskParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto src = param.Y;
    auto attn = param.X;
    auto src_offset = src->lod()[0];
    auto attn_offset = attn->lod()[0];
    int attn_seq_num = attn_offset.size() - 1;
    int src_seq_num = src_offset.size() - 1;
    int attn_seq_len = attn_offset[1];
    int src_seq_len = attn->numel() / attn->dims()[0];
    size_t count = attn->numel();
    auto attn_data = attn->data<T>();

    auto out = param.Out;
    out->Resize(attn->dims());
    out->set_lod(attn->lod());
    auto out_data = out->mutable_data<T>();
    memcpy(out_data, attn_data, count * sizeof(T));

    for (int i = 0; i < attn_seq_num; ++i) {
      for (int j = 0; j < attn_seq_len; ++j) {
        auto tmp_out_data = out_data + src_seq_len * (attn_seq_len * i + j);
        int src_seq_idx = i % src_seq_num;
        int cur_len = src_offset[src_seq_idx + 1] - src_offset[src_seq_idx];
        for (int k = cur_len; k < src_seq_len; k++) {
          tmp_out_data[k] = param.mask;
        }
      }
    }
  }

  virtual ~AttentionPaddingMaskCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
