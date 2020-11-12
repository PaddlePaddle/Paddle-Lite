// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/sequence_pad_op.h"
#include <algorithm>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequencePadOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.PadValue);
  CHECK_OR_FALSE(param_.Out);
  CHECK_OR_FALSE(param_.Length);

  return true;
}

bool SequencePadOp::InferShapeImpl() const {
  auto x_dims = param_.X->dims();
  CHECK_GE(x_dims.size(), 2) << "The rank of SequencePad OP Input(x) can't be "
                                "less than 2. But the rank we received is "
                             << x_dims.size();
  auto time_step_dims = x_dims.Slice(1, x_dims.size());
  auto pad_value_dims = param_.PadValue->dims();
  CHECK_EQ((pad_value_dims == DDim({1})) || (pad_value_dims == time_step_dims),
           true)
      << "The SequencePad OP Input(PadValue) must be a scalar or a tensor "
         "whiose shape equals to time steps in sequences";

  auto x_lod = param_.X->lod();
  CHECK_EQ(x_lod.empty(), false)
      << "The SequencePad OP Input(X) must hold lod info.";
  const auto &x_lod_0 = x_lod[0];
  CHECK_GE(x_lod_0.size(), 2)
      << "The size of SequencePadOp Input(X)'s lod info can't be less than 2. "
         "But the size we received is "
      << x_lod_0.size();
  CHECK_EQ(x_dims[0], static_cast<int64_t>(x_lod_0.back()))
      << "The SequencePadOp Input(X)'s lod info mismatches the actual tensor "
         "shape. The 1st dimension of Input(X)'s lod info is "
      << x_dims[0] << ", the 1st dimension of actual tensor shape is "
      << static_cast<int64_t>(x_lod_0.back());

  int seq_num = x_lod_0.size() - 1;
  int max_seq_len = 0;
  for (int i = 0; i < seq_num; ++i) {
    max_seq_len =
        (std::max)(max_seq_len, static_cast<int>(x_lod_0[i + 1] - x_lod_0[i]));
  }
  int real_padded_length = param_.padded_length;
  if (real_padded_length == -1) {
    real_padded_length = max_seq_len;
  }
  CHECK_GE(real_padded_length, max_seq_len)
      << "The SequencePadOp Attr(padded_length) should be greater than or "
         "equal to the length of the longest original sequence. But the "
         "padded_length we received is "
      << real_padded_length
      << ", the length of the longest original sequence is " << max_seq_len;

  int out_dim_0 = seq_num;
  std::vector<int64_t> out_dims_vec{out_dim_0, real_padded_length};
  std::vector<int64_t> len_dims_vec{out_dim_0};
  auto time_step_dims_vec = time_step_dims.Vectorize();
  out_dims_vec.insert(
      out_dims_vec.end(), time_step_dims_vec.begin(), time_step_dims_vec.end());
  param_.Out->Resize(out_dims_vec);
  param_.Length->Resize(len_dims_vec);
  return true;
}

bool SequencePadOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.PadValue = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("PadValue").front())->Get<lite::Tensor>());
  param_.Length = scope->FindVar(opdesc.Output("Length").front())
                      ->GetMutable<lite::Tensor>();
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  param_.padded_length = opdesc.GetAttr<int>("padded_length");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_pad, paddle::lite::operators::SequencePadOp);
