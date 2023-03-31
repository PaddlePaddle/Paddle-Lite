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

#include "lite/operators/__xpu__token_scatter_op.h"
#include <cmath>  // std::sqrt
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUTokenScatterOp::CheckShape() const {
  CHECK(param_.CLSInds != nullptr);
  CHECK(param_.X != nullptr);
  CHECK(param_.Updates != nullptr);
  CHECK(param_.Out != nullptr);
  return true;
}

bool XPUTokenScatterOp::InferShapeImpl() const {
  param_.Out->Resize(param_.X->dims());
  return true;
}

bool XPUTokenScatterOp::AttachImpl(const cpp::OpDesc& op_desc,
                                   lite::Scope* scope) {
  param_.CLSInds =
      &scope->FindVar(op_desc.Input("CLSInds").front())->Get<lite::Tensor>();
  param_.X = &scope->FindVar(op_desc.Input("X").front())->Get<lite::Tensor>();
  param_.Updates =
      &scope->FindVar(op_desc.Input("Updates").front())->Get<lite::Tensor>();
  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<lite::Tensor>();
  if (op_desc.HasInput("SeqLod") && op_desc.HasInput("PadSeqLen")) {
    param_.SeqLod =
        &scope->FindVar(op_desc.Input("SeqLod").front())->Get<lite::Tensor>();
    param_.PadSeqLen = &scope->FindVar(op_desc.Input("PadSeqLen").front())
                            ->Get<lite::Tensor>();
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__token_scatter,
                 paddle::lite::operators::XPUTokenScatterOp);
