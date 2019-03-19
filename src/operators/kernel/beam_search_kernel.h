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

#ifdef BEAM_SEARCH_OP

#pragma once

#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

#define GET_VAR_AS_LOD_TENSOR(name, name_dict, scope) \
  OpParam::GetVarValue<framework::LoDTensor>(name, name_dict, scope)

template <typename Dtype>
class BeamSearchParam : public OpParam {
 public:
  BeamSearchParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                  const AttributeMap &attrs, Scope *scope)
      : OpParam(inputs, outputs, attrs, scope) {
    pre_ids_ = GET_VAR_AS_LOD_TENSOR("pre_ids", inputs, *scope);
    pre_scores_ = GET_VAR_AS_LOD_TENSOR("pre_scores", inputs, *scope);
    ids_ = GET_VAR_AS_LOD_TENSOR("ids", inputs, *scope);
    scores_ = GET_VAR_AS_LOD_TENSOR("scores", inputs, *scope);

    selected_ids_ = GET_VAR_AS_LOD_TENSOR("selected_ids", outputs, *scope);
    selected_scores_ =
        GET_VAR_AS_LOD_TENSOR("selected_scores", outputs, *scope);
    if (outputs.count("parent_idx")) {
      parent_idx_ = GET_VAR_AS_LOD_TENSOR("parent_idx", outputs, *scope);
    } else {
      parent_idx_ = new framework::Tensor();
    }

    level_ = OpParam::GetAttr<int>("level", attrs);
    beam_size_ = OpParam::GetAttr<int>("beam_size", attrs);
    end_id_ = OpParam::GetAttr<int>("end_id", attrs);
    if (OpParam::HasAttr("is_accumulated", attrs)) {
      is_accumulated_ = OpParam::GetAttr<bool>("is_accumulated", attrs);
    }
  }

 public:
  framework::LoDTensor *pre_ids_;
  framework::LoDTensor *pre_scores_;
  framework::LoDTensor *ids_;
  framework::LoDTensor *scores_;

  framework::LoDTensor *selected_ids_;
  framework::LoDTensor *selected_scores_;
  framework::Tensor *parent_idx_;

  int level_;
  int beam_size_;
  int end_id_;
  bool is_accumulated_ = true;
};

DECLARE_KERNEL(BeamSearch, BeamSearchParam);

}  // namespace operators
}  // namespace paddle_mobile

#endif  // BEAM_SEARCH_OP
