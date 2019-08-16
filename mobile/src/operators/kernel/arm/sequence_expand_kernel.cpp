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

#ifdef SEQUENCE_EXPAND_OP

#include <vector>
#include "operators/kernel/sequence_kernels.h"

namespace paddle_mobile {
namespace operators {

typedef int (*LoDElementFunctor)(const std::vector<size_t> &x_lod, int index);

int element_with_lod(const std::vector<size_t> &x_lod, int index) {
  return x_lod[index];
}

int element_without_lod(const std::vector<size_t> &x_lod, int index) {
  return index;
}

template <typename T>
inline void SequenceExpandImpl(const framework::LoDTensor &x,
                               const std::vector<size_t> &ref_lod,
                               framework::LoDTensor *output) {
  const T *x_data = x.data<T>();
  auto &x_lod = x.lod();
  LoDElementFunctor lod_element = element_without_lod;
  if (x_lod.size() == 1) lod_element = element_with_lod;

  T *output_data = output->mutable_data<T>();
  int x_item_length = x.numel() / x.dims()[0];
  int out_offset = 0;

  for (size_t i = 1; i < ref_lod.size(); ++i) {
    int repeat_num = ref_lod[i] - ref_lod[i - 1];
    int x_start = lod_element(x_lod[0], i - 1);
    int x_end = lod_element(x_lod[0], i);
    int x_seq_len = x_end - x_start;
    if (repeat_num > 0) {
      int out_start = out_offset;
      if (output->lod().size() == 1) {
        out_start = output->lod()[0][out_offset];
      }
      for (int j = 0; j < repeat_num; j++) {
        for (int k = 0; k < x_seq_len; k++) {
          memcpy(output_data + (out_start + j * x_seq_len + k) * x_item_length,
                 x_data + (x_start + k) * x_item_length,
                 x_item_length * sizeof(T));
        }
      }
    }
    out_offset += repeat_num;
  }
}

template <typename T>
class SequenceExpandKernel<CPU, T>
    : public framework::OpKernelBase<CPU, SequenceExpandParam<CPU>> {
 public:
  bool Init(SequenceExpandParam<CPU> *param) { return true; }

  void Compute(const SequenceExpandParam<CPU> &param) {
    const framework::LoDTensor *input_x = param.input_x_;
    const framework::LoDTensor *input_y = param.input_y_;
    framework::LoDTensor *output = param.output_;
    output->mutable_data<T>();

    const auto &x_lod = input_x->lod();
    const auto &y_lod = input_y->lod();
    int ref_level = param.ref_level_;
    if (ref_level == -1) ref_level = y_lod.size() - 1;

    if (y_lod[ref_level].size() <= 1) {
      framework::TensorCopy(*input_x, output);
      output->set_lod(input_x->lod());
      return;
    }

    std::vector<size_t> out_lod;
    if (x_lod.size() == 1) {
      out_lod.push_back(0);
      for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
        int repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1];
        int x_start = x_lod[0][i - 1];
        int x_end = x_lod[0][i];
        int x_seq_len = x_end - x_start;
        for (int j = 0; j < repeat_num; ++j) {
          out_lod.push_back(out_lod.back() + x_seq_len);
        }
      }
      output->set_lod({out_lod});
    }
    SequenceExpandImpl<T>(*input_x, y_lod[ref_level], output);
  }
};

template class SequenceExpandKernel<CPU, float>;
// template class SequenceExpandKernel<CPU, int64_t>;

}  // namespace operators
}  // namespace paddle_mobile

#endif  // SEQUENCE_EXPAND_OP
