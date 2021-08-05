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
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
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
    auto* bottom0 = param.X;
    auto* bottom1 = param.Y;
    auto* _pad_begin = param.pad_begin;
    auto* top = param.Out;
    int _pad_id = param.pad_id;
    float _mask = param.mask;
    auto src_len = static_cast<int64_t>(bottom1->lod()[0][1]);
    const int att_batch = bottom0->lod()[0].size() - 1;
    const int src_batch = bottom1->lod()[0].size() - 1;
    int* pad_begin = _pad_begin->template mutable_data<int>();
    for (int i = 0; i < src_batch; ++i) {
      const auto* src_data = bottom1->template data<T>() + src_len * i;
      int index = src_len - 1;
      for (; index >= 0 && _pad_id == static_cast<int>(src_data[index]);
           --index) {
      }
      pad_begin[i] = index + 1;
    }

    const auto att_len = static_cast<int64_t>(bottom0->lod()[0][1]);
    auto* top_data = top->template mutable_data<T>();
    memcpy(top_data,
           bottom0->template data<T>(),
           bottom0->dims()[0] * bottom0->dims()[1] * sizeof(T));
    for (int i = 0; i < att_batch; ++i) {
      for (int j = 0; j < att_len; ++j) {
        top_data =
            top->template mutable_data<T>() + src_len * (att_len * i + j);
        int src_idx = i % src_batch;
        for (int k = pad_begin[src_idx]; k < src_len; ++k) {
          top_data[k] = _mask;
        }
      }
    }
  }

  virtual ~AttentionPaddingMaskCompute() = default;

 private:
  lite::Tensor src_offset_;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
