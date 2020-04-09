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
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SearchGroupPaddingCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SearchGroupPaddingParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::SearchGroupPaddingParam>();

    auto* bottom0 = param.x;
    auto* top0 = param.out_emb_padding;
    auto* top1 = param.out_new;
    auto* top2 = param.out_padding;

    int _pad_id = param.pad_id;

    int batch = bottom0->lod()[0].size() - 1;
    int dim0 = bottom0->dims()[0];
    int dim1 = bottom0->dims()[1];

    const auto offset = bottom0->lod()[0];
    int max_seq = 0;
    for (int i = 0; i < batch; ++i) {
      if (offset[i + 1] - offset[i] > max_seq) {
        max_seq = offset[i + 1] - offset[i];
      }
    }

    std::vector<uint64_t> new_offset;
    new_offset.resize(batch + 1);
    for (int i = 0; i < batch + 1; ++i) {
      new_offset[i] = i * max_seq;
    }

    // for padding data
    lite::LoD top0_lod;
    top0_lod.push_back(new_offset);
    top0->set_lod(top0_lod);
    top0->Resize({batch * max_seq, dim1});
    // for origin input id
    // already set by ShareLoD in InferShape
    lite::LoD top1_lod;
    top1_lod.push_back(offset);
    top1->set_lod(top1_lod);
    top1->Resize({dim0, 1});
    memset(top1->template mutable_data<T>(),
           0,
           top1->dims()[0] * top1->dims()[1] * sizeof(T));
    // for padding input id
    lite::LoD top2_lod;
    top2_lod.push_back(new_offset);
    top2->set_lod(top2_lod);
    top2->Resize({batch * max_seq, 1});
    // copy data
    const auto* bottom_data = bottom0->template data<T>();
    auto* top_data = top0->template mutable_data<T>();
    auto* top_padding_input_data = top2->template mutable_data<T>();
    for (int i = 0; i < batch; i++) {
      const int copy_step = offset[i + 1] - offset[i];
      const int start = i * max_seq;
      memcpy(top_data + start * dim1,
             bottom_data + offset[i] * dim1,
             copy_step * dim1 * sizeof(T));
      memset(top_data + (start + copy_step) * dim1,
             0,
             (max_seq - copy_step) * dim1 * sizeof(T));
      // for padding input id
      memset(top_padding_input_data + start, 0, copy_step * sizeof(T));
      for (int j = start + copy_step; j < start + max_seq; j++) {
        top_padding_input_data[j] = static_cast<T>(_pad_id);
      }
    }
  }

  virtual ~SearchGroupPaddingCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
