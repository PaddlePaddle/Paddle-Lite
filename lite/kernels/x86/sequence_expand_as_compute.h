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

#include <string>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
#include "lite/fluid/eigen.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

using Tensor = lite::Tensor;

template <typename T>
struct SequenceExpandFunctor {
  void operator()(
      const Tensor &x,
      const std::vector<uint64_t> &ref_lod, /*expand referenced lod*/
      Tensor *out) {
    int64_t hight = x.dims()[0];
    int64_t width = x.data_size() / hight;

    const T *in_data = x.data<T>();
    T *out_data = out->mutable_data<T, T>();

    for (int h_id = 0; h_id < hight; ++h_id) {
      uint64_t span = ref_lod[h_id + 1] - ref_lod[h_id];
      if (span == 0) continue;
      const T *src = in_data + h_id * width;
      for (uint64_t w_id = 0; w_id < width; ++w_id) {
        T ele = src[w_id];
        size_t offset = ref_lod[h_id] * width;
        for (uint64_t k = 0; k < span; ++k) {
          out_data[offset + k * width + w_id] = ele;
        }
      }
    }
  }
};

template <typename T>
class SequenceExpandAsCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto &param = *param_.get_mutable<operators::SequenceExpandAsParam>();

    auto *x = param.x;
    auto *y = param.y;
    auto *out = param.out;

    auto &y_lod = y->lod();
    CHECK_EQ(y_lod.size(), 1u);
    CHECK_GT(y_lod[0].size(), 1u);

    out->template mutable_data<T, T>();

    SequenceExpandFunctor<T> seq_espand_functor;
    seq_espand_functor(*x, y_lod[0], out);
  }
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
