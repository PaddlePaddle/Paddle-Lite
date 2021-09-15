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
inline LoD ConcatLoD(const std::vector<lite::Tensor*>& xs,
                     std::vector<lite::Tensor>* xs_in_order) {
  std::vector<uint64_t> result;
  result.resize(xs[0]->lod()[0].size());

  for (size_t i = 1; i < result.size(); ++i) {
    size_t sum = 0;
    for (size_t j = 0; j < xs.size(); ++j) {
      auto& x_lod = xs[j]->lod()[0];
      if (x_lod[i - 1] < x_lod[i]) {
        xs_in_order->emplace_back(xs[j]->Slice<T>(x_lod[i - 1], x_lod[i]));
      }
      sum += x_lod[i];
    }
    result[i] = sum;
  }
  LoD lod;
  lod.emplace_back(result);
  return lod;
}

template <typename T>
class SequenceConcatCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SequenceConcatParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();

    int64_t batch_size = 0;
    int64_t feature_size = 0;
    std::vector<int64_t> out_dims;
    for (const auto& tensor : param.X) {
      const auto x_dims = tensor->dims();
      CHECK(x_dims[0]);
      if (out_dims.empty()) {
        out_dims = x_dims.Vectorize();
      }
      batch_size += x_dims[0];
      if (feature_size == 0) {
        feature_size = x_dims.production() / x_dims[0];
      } else {
        CHECK_EQ(feature_size, x_dims.production() / x_dims[0])
            << "Inputs of sequence concat must have same feature size";
      }
    }
    if (batch_size < 0) {
      batch_size = -1;  // Normalize batch size for compile time.
    }
    out_dims[0] = batch_size;
    param.Out->Resize(out_dims);

    T* dout = param.Out->template mutable_data<T>();

    std::vector<lite::Tensor> x_in_order;
    param.Out->set_lod(ConcatLoD<T>(param.X, &x_in_order));

    int num = x_in_order.size();
    int out_rows = 1;

    std::vector<int64_t> input_cols(num);
    for (int i = 0; i < num; ++i) {
      input_cols[i] = x_in_order[i].numel() / out_rows;
    }

    int col_idx = 0;
    for (int j = 0; j < num; ++j) {
      int col_len = input_cols[j];
      auto input_data = x_in_order[j].data<T>();
      memcpy(dout + col_idx, input_data, sizeof(T) * col_len);
      col_idx += col_len;
    }
  }

  virtual ~SequenceConcatCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
