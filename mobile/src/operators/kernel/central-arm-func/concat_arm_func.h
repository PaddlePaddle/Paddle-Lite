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

#ifdef CONCAT_OP
#pragma once

#include <vector>

namespace paddle_mobile {
namespace operators {
template <typename T>
class ConcatFunctor {
 public:
  void operator()(const std::vector<framework::Tensor> &input, const int axis,
                  framework::Tensor *output) {
    size_t num = input.size();
    int rows = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int out_rows = rows, out_cols = 0;

    std::vector<int64_t> input_cols(input.size());
    for (int i = 0; i < num; ++i) {
      int t_cols = input[i].numel() / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }

    // computation
    for (int k = 0; k < out_rows; ++k) {
      T *dst_ptr = output->data<T>() + k * out_cols;
      int col_idx = 0;
      for (int j = 0; j < num; ++j) {
        int col_len = input_cols[j];
        const T *src_prt = input[j].data<T>() + k * col_len;
        memory::Copy(dst_ptr + col_idx, src_prt, sizeof(T) * col_len);
        col_idx += col_len;
      }
    }
  }
};

template <typename P>
void ConcatCompute(const ConcatParam<CPU> &param) {
  auto inputs = param.Inputs();
  auto *out = param.Out();
  int axis = param.Axis();
  out->mutable_data<P>();

  /// Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && inputs.size() < 10) {
    size_t output_offset = 0;
    for (auto *in : inputs) {
      auto in_stride = framework::stride_numel(in->dims());
      auto out_stride = framework::stride_numel(out->dims());
      auto dst = out->data<P>() + output_offset;
      auto src = in->data<P>();
      PADDLE_MOBILE_ENFORCE(
          in_stride.size() == out_stride.size(),
          "src and dst tensor should have the same dims size.");
      memory::Copy(dst, src, sizeof(P) * in_stride[0]);
      output_offset += in_stride[0];
    }
  } else {
    std::vector<framework::Tensor> inputs_concat(inputs.size());
    for (int j = 0; j < inputs.size(); ++j) {
      inputs_concat[j] = *inputs[j];
    }
    ConcatFunctor<P> concat_functor;
    concat_functor(inputs_concat, axis, out);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
