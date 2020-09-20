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
#include <algorithm>
#include <cstring>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SequenceArithmeticCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SequenceArithmeticParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto x = param.X;
    auto y = param.Y;
    auto out = param.Out;
    int op_type = param.op_type;

    out->Resize(x->dims());
    out->set_lod(x->lod());

    auto x_data = x->template data<T>();
    auto y_data = y->template data<T>();
    auto out_data = out->template mutable_data<T>();
    auto x_seq_offset = x->lod()[0];
    auto y_seq_offset = y->lod()[0];
    int seq_num = x_seq_offset.size() - 1;
    int inner_size = (x->numel()) / (x->dims()[0]);

    // sum
    if (op_type == 1) {
      for (int i = 0; i < seq_num; i++) {
        int len_x = (x_seq_offset[i + 1] - x_seq_offset[i]) * inner_size;
        int len_y = (y_seq_offset[i + 1] - y_seq_offset[i]) * inner_size;
        auto input_x = x_data + x_seq_offset[i] * inner_size;
        auto input_y = y_data + y_seq_offset[i] * inner_size;
        auto t_out = out_data + x_seq_offset[i] * inner_size;
        int len = (std::min)(len_x, len_y);
        for (int j = 0; j < len; j++) {
          t_out[j] = input_x[j] + input_y[j];
        }
        if (len_x > len) {
          memcpy(t_out + len, input_x + len, sizeof(T) * (len_x - len));
        }
      }
    }

    // sub
    if (op_type == 2) {
      for (int i = 0; i < seq_num; i++) {
        int len_x = (x_seq_offset[i + 1] - x_seq_offset[i]) * inner_size;
        int len_y = (y_seq_offset[i + 1] - y_seq_offset[i]) * inner_size;
        auto input_x = x_data + x_seq_offset[i] * inner_size;
        auto input_y = y_data + y_seq_offset[i] * inner_size;
        auto t_out = out_data + x_seq_offset[i] * inner_size;
        int len = (std::min)(len_x, len_y);
        for (int j = 0; j < len; j++) {
          t_out[j] = input_x[j] - input_y[j];
        }
        if (len_x > len) {
          memcpy(t_out + len, input_x + len, sizeof(T) * (len_x - len));
        }
      }
    }

    // mul
    if (op_type == 3) {
      for (int i = 0; i < seq_num; i++) {
        int len_x = (x_seq_offset[i + 1] - x_seq_offset[i]) * inner_size;
        int len_y = (y_seq_offset[i + 1] - y_seq_offset[i]) * inner_size;
        auto input_x = x_data + x_seq_offset[i] * inner_size;
        auto input_y = y_data + y_seq_offset[i] * inner_size;
        auto t_out = out_data + x_seq_offset[i] * inner_size;
        int len = (std::min)(len_x, len_y);
        for (int j = 0; j < len; j++) {
          t_out[j] = input_x[j] * input_y[j];
        }
        if (len_x > len) {
          memcpy(t_out + len, input_x + len, sizeof(T) * (len_x - len));
        }
      }
    }
  }

  virtual ~SequenceArithmeticCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
