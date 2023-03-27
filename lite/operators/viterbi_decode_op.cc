// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/viterbi_decode_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ViterbiDecodeOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.length);
  CHECK_OR_FALSE(param_.transition);
  CHECK_OR_FALSE(param_.path);
  CHECK_OR_FALSE(param_.scores);
  return true;
}

bool ViterbiDecodeOpLite::InferShapeImpl() const { return true; }

bool ViterbiDecodeOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                     lite::Scope *scope) {
  CHECK(!op_desc.Input("Input").empty());
  CHECK(!op_desc.Input("Length").empty());
  CHECK(!op_desc.Input("Transition").empty());
  CHECK(!op_desc.Output("Path").empty());
  CHECK(!op_desc.Output("Scores").empty());
  auto Input = op_desc.Input("Input").front();
  auto Length = op_desc.Input("Length").front();
  auto Transition = op_desc.Input("Transition").front();
  auto Path = op_desc.Output("Path").front();
  auto Scores = op_desc.Output("Scores").front();
  param_.input = GetVar<lite::Tensor>(scope, Input);
  param_.length = GetVar<lite::Tensor>(scope, Length);
  param_.transition = GetVar<lite::Tensor>(scope, Transition);
  param_.scores = GetMutableVar<lite::Tensor>(scope, Scores);
  param_.path = GetMutableVar<lite::Tensor>(scope, Path);
  param_.include_bos_eos_tag = op_desc.GetAttr<bool>("include_bos_eos_tag");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(viterbi_decode, paddle::lite::operators::ViterbiDecodeOpLite);
