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

#include "operators/sequence_ops/sequence_expand_op.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
void SequenceExpandOp<DeviceType, T>::InferShape() const {
  const auto *input_x = this->param_.input_x_;
  const auto *input_y = this->param_.input_y_;
  const auto &x_lod = input_x->lod();
  const auto &y_lod = input_y->lod();
  int ref_level = this->param_.ref_level_;
  if (ref_level == -1) ref_level = y_lod.size() - 1;

  auto out_dims = input_x->dims();
  int64_t out_first_dim = 0;

  if (y_lod[ref_level].size() > 1) {
    for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
      int x_seq_len = 1;
      if (x_lod.size() == 1) {
        x_seq_len = x_lod[0][i] - x_lod[0][i - 1];
      }
      out_first_dim +=
          (y_lod[ref_level][i] - y_lod[ref_level][i - 1]) * x_seq_len;
    }
    out_dims[0] = out_first_dim;
  }
  this->param_.output_->Resize(out_dims);
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(sequence_expand, ops::SequenceExpandOp);
#endif

#endif  // SEQUENCE_EXPAND_OP
