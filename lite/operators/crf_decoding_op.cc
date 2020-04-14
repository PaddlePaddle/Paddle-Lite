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

#include "lite/operators/crf_decoding_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CrfDecodingOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.emission);
  CHECK_OR_FALSE(param_.transition);
  CHECK_OR_FALSE(param_.viterbi_path);

  auto emission_dims = param_.emission->dims();
  if (param_.length == nullptr) {
    CHECK_OR_FALSE(emission_dims.size() == 2);
  } else {
    CHECK_OR_FALSE(emission_dims.size() == 3);
  }
  CHECK_OR_FALSE(emission_dims[0] != 0);

  auto transition_dims = param_.transition->dims();
  CHECK_OR_FALSE(transition_dims.size() == 2);
  CHECK_OR_FALSE(transition_dims[0] - 2 == transition_dims[1]);

  if ((emission_dims[emission_dims.size() - 1] > 0 &&
       transition_dims[transition_dims.size() - 1] > 0)) {
    CHECK_OR_FALSE(emission_dims[emission_dims.size() - 1] ==
                   transition_dims[transition_dims.size() - 1]);
  }

  if (param_.label != nullptr) {
    auto label_dims = param_.label->dims();
    if (param_.length != nullptr) {
      CHECK_OR_FALSE((label_dims.size() == 3UL && label_dims[2] == 1) ||
                     label_dims.size() == 2UL);
    } else {
      CHECK_OR_FALSE((label_dims.size() == 2UL && label_dims[1] == 1) ||
                     label_dims.size() == 1UL);
    }
    if (emission_dims[0] > 0 && label_dims[0] > 0) {
      CHECK_OR_FALSE(emission_dims[0] == label_dims[0]);
    }
  }
  return true;
}

bool CrfDecodingOpLite::InferShapeImpl() const {
  auto emission_dims = param_.emission->dims();
  if (param_.length == nullptr) {
    param_.viterbi_path->Resize({emission_dims[0], 1});
  } else {
    param_.viterbi_path->Resize({emission_dims[0], emission_dims[1]});
  }
  param_.viterbi_path->set_lod(param_.emission->lod());
  return true;
}

bool CrfDecodingOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                   lite::Scope *scope) {
  // inputs
  param_.emission = scope->FindVar(op_desc.Input("Emission").front())
                        ->GetMutable<lite::Tensor>();
  param_.transition = scope->FindVar(op_desc.Input("Transition").front())
                          ->GetMutable<lite::Tensor>();
  if (op_desc.HasInput("Label") && op_desc.Input("Label").size() > 0) {
    param_.label = scope->FindVar(op_desc.Input("Label").front())
                       ->GetMutable<lite::Tensor>();
  }
  if (op_desc.HasInput("Length") && op_desc.Input("Length").size() > 0) {
    param_.length = scope->FindVar(op_desc.Input("Length").front())
                        ->GetMutable<lite::Tensor>();
  }

  // outputs
  param_.viterbi_path = scope->FindVar(op_desc.Output("ViterbiPath").front())
                            ->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(crf_decoding, paddle::lite::operators::CrfDecodingOpLite);
