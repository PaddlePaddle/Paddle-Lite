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

#include "lite/operators/beam_search_decode_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool BeamSearchDecodeOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.ids)
  CHECK_OR_FALSE(param_.scores)
  CHECK_OR_FALSE(param_.sentence_ids)
  CHECK_OR_FALSE(param_.sentence_scores)
  return true;
}

bool BeamSearchDecodeOpLite::InferShapeImpl() const { return true; }

bool BeamSearchDecodeOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                        lite::Scope *scope) {
  auto ids = op_desc.Input("Ids").front();
  auto scores = op_desc.Input("Scores").front();
  auto sentence_ids = op_desc.Output("SentenceIds").front();
  auto sentence_scores = op_desc.Output("SentenceScores").front();

  param_.ids = scope->FindVar(ids)->GetMutable<std::vector<lite::Tensor>>();
  param_.scores =
      scope->FindVar(scores)->GetMutable<std::vector<lite::Tensor>>();
  param_.sentence_ids = scope->FindMutableTensor(sentence_ids);
  param_.sentence_scores = scope->FindMutableTensor(sentence_scores);

  param_.beam_size = op_desc.GetAttr<int>("beam_size");
  param_.end_id = op_desc.GetAttr<int>("end_id");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(beam_search_decode,
                 paddle::lite::operators::BeamSearchDecodeOpLite)
