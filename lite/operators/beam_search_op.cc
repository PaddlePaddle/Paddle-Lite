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

#include "lite/operators/beam_search_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool BeamSearchOp::CheckShape() const {
  CHECK_OR_FALSE(param_.pre_ids);
  CHECK_OR_FALSE(param_.pre_scores);
  CHECK_OR_FALSE(param_.ids);
  CHECK_OR_FALSE(param_.scores);
  CHECK_OR_FALSE(param_.selected_ids);
  CHECK_OR_FALSE(param_.selected_scores);
  CHECK_OR_FALSE(param_.parent_idx);
  return true;
}

bool BeamSearchOp::InferShape() const { return true; }

bool BeamSearchOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.pre_ids = scope->FindVar(opdesc.Input("pre_ids").front())
                       ->GetMutable<lite::Tensor>();
  param_.pre_scores = scope->FindVar(opdesc.Output("pre_scores").front())
                          ->GetMutable<lite::Tensor>();
  param_.ids =
      scope->FindVar(opdesc.Output("ids").front())->GetMutable<lite::Tensor>();
  param_.scores = scope->FindVar(opdesc.Output("scores").front())
                      ->GetMutable<lite::Tensor>();
  param_.selected_ids = scope->FindVar(opdesc.Output("selected_ids").front())
                            ->GetMutable<lite::Tensor>();
  param_.selected_scores =
      scope->FindVar(opdesc.Output("selected_scores").front())
          ->GetMutable<lite::Tensor>();
  param_.parent_idx = scope->FindVar(opdesc.Output("parent_idx").front())
                          ->GetMutable<lite::Tensor>();
  CHECK(param_.pre_ids);
  CHECK(param_.pre_scores);
  CHECK(param_.ids);
  CHECK(param_.scores);
  CHECK(param_.selected_ids);
  CHECK(param_.selected_scores);
  CHECK(param_.parent_idx);
  param_.level = opdesc.GetAttr<int>("level");
  param_.beam_size = opdesc.GetAttr<int>("beam_size");
  param_.end_id = opdesc.GetAttr<int>("end_id");
  param_.is_accumulated = opdesc.GetAttr<bool>("is_accumulated");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(beam_search, paddle::lite::operators::BeamSearchOp);
