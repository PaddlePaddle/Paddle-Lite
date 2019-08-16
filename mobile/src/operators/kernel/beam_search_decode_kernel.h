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

#ifdef BEAM_SEARCH_DECODE_OP

#pragma once

#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename Dtype>
class BeamSearchDecodeParam : public OpParam {
 public:
  BeamSearchDecodeParam(const VariableNameMap &inputs,
                        const VariableNameMap &outputs,
                        const AttributeMap &attrs, Scope *scope)
      : OpParam(inputs, outputs, attrs, scope) {
    ids_ =
        OpParam::GetVarValue<framework::LoDTensorArray>("Ids", inputs, *scope);
    scores_ = OpParam::GetVarValue<framework::LoDTensorArray>("Scores", inputs,
                                                              *scope);
    sentence_ids_ = OpParam::GetVarValue<framework::LoDTensor>("SentenceIds",
                                                               outputs, *scope);
    sentence_scores_ = OpParam::GetVarValue<framework::LoDTensor>(
        "SentenceScores", outputs, *scope);
    beam_size_ = OpParam::GetAttr<int>("beam_size", attrs);
    end_id_ = OpParam::GetAttr<int>("end_id", attrs);
  }

 public:
  framework::LoDTensorArray *ids_;
  framework::LoDTensorArray *scores_;
  framework::LoDTensor *sentence_ids_;
  framework::LoDTensor *sentence_scores_;
  int beam_size_;
  int end_id_;
};

DECLARE_KERNEL(BeamSearchDecode, BeamSearchDecodeParam);

}  // namespace operators
}  // namespace paddle_mobile

#endif  // BEAM_SEARCH_DECODE_OP
