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

#ifdef TRANSPOSE2_OP

#include "operators/kernel/transpose2_kernel.h"

namespace paddle_mobile {
namespace operators {

bool IsShuffleChannel(const std::vector<int> &axis) {
  bool is_shuffle_channel = true;
  if (axis.size() > 2 && axis[0] == 0 && axis[1] == 2 && axis[2] == 1) {
    for (int i = 3; i < axis.size(); ++i) {
      if (axis[i] != i) {
        is_shuffle_channel = false;
        break;
      }
    }
  } else {
    return false;
  }
  return is_shuffle_channel;
}

template <typename Dtype>
void ShuffleChannelCompute(const Transpose2Param<CPU> &param) {
  const std::vector<int> &axis = param.Axis();
  const Tensor *input = param.InputX();
  const Dtype *input_ptr = input->data<Dtype>();
  Tensor *output = param.Out();
  Dtype *output_ptr = output->mutable_data<Dtype>();
  // input and output's shape dimension must >= 2 && <= 6.
  const framework::DDim &in_dim = input->dims();
  const framework::DDim &out_dim = output->dims();
  size_t offset = 1;
  for (int i = 3; i < axis.size(); ++i) {
    offset *= in_dim[i];
  }

  #pragma omp parallel for collapse(3)
  for (int batch = 0; batch < out_dim[0]; ++batch) {
    for (int c1 = 0; c1 < out_dim[1]; ++c1) {
      for (int c2 = 0; c2 < out_dim[2]; ++c2) {
        size_t out_offset =
            ((batch * out_dim[1] + c1) * out_dim[2] + c2) * offset;
        size_t in_offset = ((batch * in_dim[1] + c2) * in_dim[2] + c1) * offset;
        memcpy(output_ptr + out_offset, input_ptr + in_offset,
               offset * sizeof(Dtype));
      }
    }
  }
}

template <typename Dtype>
void Transpose2Compute(const Transpose2Param<CPU> &param) {
  const std::vector<int> &axis = param.Axis();
  const Tensor *input = param.InputX();
  const Dtype *input_ptr = input->data<Dtype>();
  Tensor *output = param.Out();
  Dtype *output_ptr = output->mutable_data<Dtype>();
  // input and output's shape dimension must >= 2 && <= 6.
  const framework::DDim &in_dim = input->dims();
  const framework::DDim &out_dim = output->dims();

  // precompute inverted output dim and strides
  size_t rout_dim[6], strides[6];
  int permute = axis.size();  // permute must >=2 && <= 6.
  for (int i = 0; i < permute; ++i) {
    int k = permute - 1 - i;
    strides[k] = 1;
    for (int j = axis[i] + 1; j < permute; ++j) {
      strides[k] *= in_dim[j];
    }
    rout_dim[k] = out_dim[i];
  }
  // unroll the first 2 dimensions
  int reamin_dim = 1;
  for (int i = 2; i < out_dim.size(); ++i) {
    reamin_dim *= out_dim[i];
  }

  #pragma omp parallel for collapse(2)
  for (int batch = 0; batch < out_dim[0]; ++batch) {
    for (int j = 0; j < out_dim[1]; ++j) {
      size_t offset = batch * strides[permute - 1] + j * strides[permute - 2];
      Dtype *out_ptr = output_ptr + (batch * out_dim[1] + j) * reamin_dim;
      int indics[4] = {0, 0, 0, 0};
      for (int k = 0; k < reamin_dim; ++k) {
        out_ptr[k] = input_ptr[offset];
        indics[0] += 1;
        offset += strides[0];
        for (int p = 0; p < permute - 3; ++p) {
          if (indics[p] == rout_dim[p]) {
            indics[p + 1] += 1;
            indics[p] = 0;
            offset += strides[p + 1];
            offset -= rout_dim[p] * strides[p];
          } else {
            break;
          }
        }
      }
    }
  }
}

template <>
bool Transpose2Kernel<CPU, float>::Init(Transpose2Param<CPU> *param) {
  return true;
}

template <>
void Transpose2Kernel<CPU, float>::Compute(const Transpose2Param<CPU> &param) {
  const std::vector<int> &axis = param.Axis();
  bool shuffle_channel = IsShuffleChannel(axis);
  if (shuffle_channel) {
    if (param.InputX()->type() == type_id<int8_t>().hash_code()) {
      ShuffleChannelCompute<int8_t>(param);
    } else {
      ShuffleChannelCompute<float>(param);
    }
  } else {
    if (param.InputX()->type() == type_id<int8_t>().hash_code()) {
      Transpose2Compute<int8_t>(param);
    } else {
      Transpose2Compute<float>(param);
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // TRANSPOSE2_OP
