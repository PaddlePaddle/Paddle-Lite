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

#include "lite/operators/ctc_align_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CtcAlignOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.input != nullptr);
  CHECK_OR_FALSE(param_.output != nullptr);

  auto* input = param_.input;
  auto* input_length = param_.input_length;
  auto input_lod = input->lod();
  CHECK_OR_FALSE(!input_lod.empty() || input_length != nullptr);
  return true;
}

bool CtcAlignOpLite::InferShapeImpl() const {
  auto input_dims = param_.input->dims();
  // It is tricky to set the wrong dimension here.
  param_.output->Resize(input_dims);
  if (param_.input_length != nullptr && param_.output_length != nullptr) {
    param_.output_length->Resize({input_dims[0], 1});
  }
  return true;
}

bool CtcAlignOpLite::AttachImpl(const cpp::OpDesc& op_desc,
                                lite::Scope* scope) {
  AttachInput(op_desc, scope, "Input", false, &param_.input);
  AttachInput(op_desc, scope, "InputLength", true, &param_.input_length);
  AttachOutput(op_desc, scope, "Output", false, &param_.output);
  AttachOutput(op_desc, scope, "OutputLength", true, &param_.output_length);
  param_.blank = op_desc.GetAttr<int>("blank");
  param_.merge_repeated = op_desc.GetAttr<bool>("merge_repeated");
  param_.padding_value = op_desc.GetAttr<int>("padding_value");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(ctc_align, paddle::lite::operators::CtcAlignOpLite);
