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
#include "lite/api/paddle_place.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
#include "lite/fluid/data_type.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

/**
 * A thin wrapper for gathering on cpu tensor
 * Return a new tensor from source tensor, gathered according to index
 * input[src]: type-T source Tensor
 * input[index]: type-IndexT index Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void CPUGather(const lite::Tensor* src,
               const lite::Tensor* index,
               lite::Tensor* output) {
  // check index of shape 1-D
  if (index->dims().size() == 2) {
    CHECK(index->dims()[1] == 1) << "Index(Input)'s dimension[1] should be 1 "
                                    "when Index(input)'s dimension's size "
                                    "equal to 2 in Gather(Op).";
  } else {
    CHECK(index->dims().size() == 1)
        << "Index(Input)'s dimension's size() should be 1 or 2 in Gather(Op).";
  }
  int64_t index_size = index->dims()[0];

  auto src_dims = src->dims();

  const T* p_src = src->template data<T>();
  const IndexT* p_index = index->data<IndexT>();
  T* p_output = output->template mutable_data<T>();

  // slice size
  int slice_size = 1;
  for (size_t i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t slice_bytes = slice_size * sizeof(T);
  for (int64_t i = 0; i < index_size; ++i) {
    int index_ = p_index[i];
    memcpy(p_output + i * slice_size, p_src + index_ * slice_size, slice_bytes);
  }
}

template <typename DataType, typename IndexType>
void CPUGatherV2(const operators::GatherParam& param) {
  auto* axis_data = param.Axis->data<IndexType>();
  auto* index_data = param.Index->data<IndexType>();
  auto* input_data = param.X->data<DataType>();
  auto* out_data = param.Out->mutable_data<DataType>();

  int index_size = param.Index->numel();
  int input_size = param.X->numel();
  auto input_dim = param.X->dims();
  int axis_index = axis_data[0];
  int inner_dim_size = 1;
  int outer_dim_size = 1;
  int input_index_dim_size = input_dim[axis_index];
  for (int i = 0; i < index_size; i++) {
    CHECK_LT(index_data[i], input_index_dim_size)
        << "The element of Index must be less than the size of"
        << "dim size of axis dim";
  }
  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
  }
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
  }

  int out_index = 0;
  for (int i = 0; i < inner_dim_size; i++) {
    for (int j = 0; j < index_size; j++) {
      for (int k = 0; k < outer_dim_size; k++) {
        int index = k + index_data[j] * outer_dim_size +
                    (i * input_size / inner_dim_size);
        out_data[out_index] = input_data[index];
        out_index++;
      }
    }
  }
}

template <typename T, typename IndexT>
class GatherCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::GatherParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    if (param.Axis != nullptr) {
      CPUGatherV2<T, IndexT>(param);
      return;
    } else {
      auto x = param.X;
      auto index = param.Index;
      auto out = param.Out;

      out->template mutable_data<T>();
      if (x->dims().production() == 0) return;
      /*
      * Since there's no type defined for lite::Tensor in Paddle-Lite, then
      * convert the Index's value to float which must be int32_t or int64_t and
      * this supposes to cause no precision difference during inference just for
      * now.
      * Alternatively, if define the Tensor's type during registering, may cause
      * a redefinition error.
      */
      CPUGather<T, IndexT>(x, index, out);
    }
  }

  virtual ~GatherCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
