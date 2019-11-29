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

#include "lite/operators/sequence_concat_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceConcatOp::CheckShape() const {
  CHECK_GT(param_.X.size(), 1)
      << "The number of input sequences is at least two.";
  CHECK_OR_FALSE(param_.Out);
  size_t lod_size = 0;
  for (const auto &t : param_.X) {
    CHECK_EQ(t->lod().empty(), false)
        << "Input Tensor of X does not contain LoD information.";
    CHECK_EQ(t->lod().size(), 1) << "Only support one level sequence now.";
    if (lod_size == 0) {
      lod_size = t->lod()[0].size();
    } else {
      CHECK_EQ(t->lod()[0].size(), lod_size)
          << "The number of sequence must be same between each input";
    }
  }
  CHECK_NE(lod_size, 0) << "Each input must have sequence information";
  return true;
}

bool SequenceConcatOp::InferShape() const {
  int64_t batch_size = 0;
  int64_t feature_size = 0;
  std::vector<int64_t> out_dims;
  for (const auto &tensor : param_.X) {
    const auto x_dims = tensor->dims();
    if (out_dims.empty()) {
      out_dims = x_dims.Vectorize();
    }
    batch_size += x_dims[0];
    if (feature_size == 0) {
      feature_size = x_dims.production() / x_dims[0];
    } else {
      CHECK_EQ(feature_size, x_dims.production() / x_dims[0])
          << "Inputs of sequence concat must have same feature size";
    }
  }
  if (batch_size < 0) {
    batch_size = -1;  // Normalize batch size for compile time.
  }
  out_dims[0] = batch_size;
  param_.Out->Resize(out_dims);
  // LoD info will be computed in Kernel.
  return true;
}

bool SequenceConcatOp::AttachImpl(const cpp::OpDesc &opdesc,
                                  lite::Scope *scope) {
  auto input_list = opdesc.Input("X");
  param_.X.clear();
  for (auto var : input_list) {
    param_.X.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  CHECK(param_.Out) << "Output(Out) of Sequence Concat Op should not be null.";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_concat, paddle::lite::operators::SequenceConcatOp);
