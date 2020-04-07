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

#include "lite/operators/sequence_unpad_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceUnpadOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Length);
  CHECK_OR_FALSE(param_.Out);
  auto x_dims = param_.X->dims();
  auto len_dims = param_.Length->dims();
  CHECK(x_dims.size() >= 2) << "Rank of X can't be less than 2";
  CHECK(len_dims.size() == 1) << "Rank of Length should be 1";
  CHECK(x_dims[0] == len_dims[0])
      << "X and Length should have the same 1st dim";
  return true;
}

bool SequenceUnpadOp::InferShapeImpl() const {
  auto x_dims = param_.X->dims();
  auto len_dims = param_.Length->dims();

  auto *seq_len_ptr = param_.Length->data<int64_t>();
  int64_t batch_size = len_dims[0];
  std::vector<uint64_t> out_lod0(batch_size + 1, 0);
  for (int64_t i = 0; i < batch_size; ++i) {
    out_lod0[i + 1] = out_lod0[i] + seq_len_ptr[i];
  }
  paddle::lite::LoD out_lod;
  out_lod.push_back(out_lod0);

  int64_t out_dim0 = out_lod0.back();
  std::vector<int64_t> out_dims{out_dim0};
  if (x_dims.size() == 2) {
    out_dims.push_back(1);
  } else {
    for (size_t i = 2; i < x_dims.size(); ++i) {
      out_dims.push_back(x_dims[i]);
    }
  }
  param_.Out->Resize(out_dims);
  param_.Out->set_lod(out_lod);
  return true;
}

bool SequenceUnpadOp::AttachImpl(const cpp::OpDesc &opdesc,
                                 lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.Length = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("Length").front())->Get<lite::Tensor>());
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_unpad, paddle::lite::operators::SequenceUnpadOp);
