/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/sequence_topk_avg_pooling.h"
#include <algorithm>
#include <vector>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <typename T>
void get_topk_pos(const T* data, int length, int k, int* pos, bool debug) {
  size_t real_k = k < length ? k : length;

  std::vector<T> v(data, data + length);

  std::vector<int> topk_pos;
  T min_val = -10000000.0;
  while (topk_pos.size() < real_k) {
    T max_val = min_val;
    int max_pos = -1;
    for (int i = 0; i < length; ++i) {
      if (v[i] > max_val) {
        max_pos = i;
        max_val = v[i];
      }
    }

    assert(max_pos >= 0);

    topk_pos.push_back(max_pos);
    v[max_pos] = min_val;
  }

  assert(topk_pos.size() > 0);
  while (topk_pos.size() < (size_t)k) {
    topk_pos.push_back(-1);
  }

  for (size_t i = 0; i < topk_pos.size(); ++i) {
    pos[i] = topk_pos[i];
  }
}

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SequenceTopkAvgPoolingFunctor<lite::TargetType::kX86, T> {
 public:
  void operator()(const lite::Tensor& in,
                  const lite::Tensor& row,
                  const lite::Tensor& col,
                  lite::Tensor* out,
                  lite::Tensor* pos,
                  int channel_num,
                  std::vector<int> topks) {
    auto k_num = topks.size();
    auto max_k = topks[topks.size() - 1];
    std::vector<int64_t> vec_pos_shape;
    auto in_lod = in.lod()[0];
    auto row_lod = row.lod()[0];
    auto col_lod = col.lod()[0];
    int batch_size = row_lod.size() - 1;
    int pos_total_size = row_lod[batch_size] * channel_num * max_k;
    vec_pos_shape.push_back(pos_total_size);
    lite::DDim dims(vec_pos_shape);
    pos->Resize(dims);
    auto pos_data = pos->mutable_data<int>(lite::TargetType::kX86);

    int offset = 0;
    std::vector<uint64_t> vec_out_lod;
    vec_out_lod.reserve(batch_size + 1);
    for (int i = 0; i <= batch_size; ++i) {
      offset = row_lod[i];
      vec_out_lod.push_back(offset);
    }

    lite::LoD lod_temp;
    lod_temp.push_back(vec_out_lod);
    out->set_lod(lod_temp);

    auto in_data = in.data<T>();
    auto out_data = out->template mutable_data<T>(lite::TargetType::kX86);

    T* sum_data = new T[max_k];
    for (int i = 0; i < batch_size; ++i) {
      int total_size = in_lod[i + 1] - in_lod[i];
      int row_size = row_lod[i + 1] - row_lod[i];
      int col_size = col_lod[i + 1] - col_lod[i];

      CHECK_EQ(total_size, channel_num * row_size * col_size)
          << "size wrong in sequence_topk_avg_pooling_op!";

      int feature_num = row_size * col_size;
      for (int j = 0; j < channel_num; ++j) {
        auto input_offset_feature_data = in_data + in_lod[i] + j * feature_num;

        for (int r = 0; r < row_size; ++r) {
          auto row_data = input_offset_feature_data + r * col_size;
          auto pos_slice_data = pos_data + row_lod[i] * channel_num * max_k +
                                r * channel_num * max_k + j * max_k;
          auto out_slice_data = out_data + row_lod[i] * channel_num * k_num +
                                r * channel_num * k_num + j * k_num;

          get_topk_pos<T>(row_data, col_size, max_k, pos_slice_data);
          if (pos_slice_data[0] == -1) {
            sum_data[0] = 0.0;
          } else {
            sum_data[0] = row_data[pos_slice_data[0]];
          }
          for (int k = 1; k < max_k; ++k) {
            if (pos_slice_data[k] == -1) {
              sum_data[k] = sum_data[k - 1];
            } else {
              sum_data[k] = sum_data[k - 1] + row_data[pos_slice_data[k]];
            }
          }
          for (size_t k = 0; k < k_num; ++k) {
            out_slice_data[k] = sum_data[topks[k] - 1] / topks[k];
          }
        }
      }
    }
    delete[] sum_data;
  }
};

#define DEFINE_FUNCTOR(type) \
  template class SequenceTopkAvgPoolingFunctor<lite::TargetType::kX86, type>;

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
