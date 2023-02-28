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

#include "lite/operators/__xpu__mask_adaptive_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUMaskAdaptiveOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Mask);
  CHECK_OR_FALSE(param_.Length);
  CHECK_OR_FALSE(param_.SeqLod);
  CHECK_OR_FALSE(param_.PadSeqLen);

  const auto mask_dims = param_.Mask->dims();
  CHECK_EQ(mask_dims.size(), 3UL) << "invalid mask dims";
  return true;
}

bool XPUMaskAdaptiveOp::InferShapeImpl() const {
  if (param_.Mask != nullptr) {
    param_.PadSeqLen->Resize({1});
  }

  return true;
}

bool XPUMaskAdaptiveOp::AttachImpl(const cpp::OpDesc& op_desc,
                                   lite::Scope* scope) {
  param_.Mask = scope->FindTensor(op_desc.Input("Mask").front());
  param_.Length = scope->FindMutableTensor(op_desc.Output("Length").front());
  param_.SeqLod = scope->FindMutableTensor(op_desc.Output("SeqLod").front());
  param_.PadSeqLen =
      scope->FindMutableTensor(op_desc.Output("PadSeqLen").front());

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__mask_adaptive,
                 paddle::lite::operators::XPUMaskAdaptiveOp);
