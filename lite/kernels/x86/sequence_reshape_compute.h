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
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SequenceReshapeCompute
    : public KernelLite<TARGET(kX86), PRECISION(kInt64)> {
 public:
  using param_t = operators::SequenceReshapeParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::SequenceReshapeParam>();
    // auto& context = context_->As<X86Context>();
    auto* in = param.x;
    auto* out = param.output;
    int out_width = param.new_dim;

    const auto& in_dims = in->dims();
    int64_t in_width = in_dims[1];

    auto& in_lod = in->lod();
    CHECK_EQ(in_lod.size(), 1UL);
    CHECK_EQ((uint64_t)in_dims[0], in_lod[0].back());

    auto in_lod_l0 = in_lod[0];
    int seq_num = in_lod_l0.size() - 1;

    if (in_width == out_width) {
      out->set_lod(in->lod());
    } else {
      auto& out_lod = *out->mutable_lod();
      out_lod.resize(1);
      out_lod[0].resize(seq_num + 1);
      out_lod[0][0] = 0;
      for (int i = 0; i < seq_num; ++i) {
        size_t seq_len = in_lod_l0[i + 1] - in_lod_l0[i];
        size_t offset = 0;
        offset = (seq_len * in_width) / out_width;
        CHECK_EQ(offset * out_width, seq_len * in_width);
        out_lod[0][i + 1] = out_lod[0][i] + offset;
      }
    }

    out->Resize(std::vector<int64_t>{in->numel() / out_width, out_width});
    auto* dst_ptr = out->template mutable_data<T>();
    auto size = in->numel() * sizeof(T);
    std::memcpy(dst_ptr, in->template data<T>(), size);
  }

  virtual ~SequenceReshapeCompute() = default;
};

template <typename T>
class SequenceReshapeFloatCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SequenceReshapeParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::SequenceReshapeParam>();
    auto* in = param.x;
    auto* out = param.output;
    auto out_data = out->template mutable_data<T>();
    for (int i = 0; i < out->dims().production(); i++) {
      out_data[i] = 0;
    }
    int out_width = param.new_dim;
    const auto& in_dims = in->dims();
    int64_t in_width = in_dims[1];
    auto& in_lod = in->lod();
    CHECK_EQ(in_lod.size(), 1UL);
    CHECK_EQ((uint64_t)in_dims[0], in_lod[0].back());
    auto in_lod_l0 = in_lod[0];
    int seq_num = in_lod_l0.size() - 1;
    if (in_width == out_width) {
      out->set_lod(in->lod());
    } else {
      auto& out_lod = *out->mutable_lod();
      out_lod.resize(1);
      out_lod[0].resize(seq_num + 1);
      out_lod[0][0] = 0;
      for (int i = 0; i < seq_num; ++i) {
        size_t seq_len = in_lod_l0[i + 1] - in_lod_l0[i];
        size_t offset = 0;
        offset = (seq_len * in_width) / out_width;
        CHECK_EQ(offset * out_width, seq_len * in_width);
        out_lod[0][i + 1] = out_lod[0][i] + offset;
      }
    }
    out->Resize(std::vector<int64_t>{in->numel() / out_width, out_width});
    auto* dst_ptr = out->template mutable_data<T>();
    auto size = in->numel() * sizeof(T);
    std::memcpy(dst_ptr, in->template data<T>(), size);
  }

  virtual ~SequenceReshapeFloatCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
