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

#include "lite/operators/box_clip_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool BoxClipOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.Input);
  CHECK_OR_FALSE(param_.ImInfo);
  CHECK_OR_FALSE(param_.Output);

  auto input_dims = param_.Input->dims();
  auto im_info_dims = param_.ImInfo->dims();
  auto input_box_size = input_dims.size();
  CHECK_OR_FALSE(input_dims[input_box_size - 1] == 4);
  CHECK_OR_FALSE(im_info_dims.size() == 2);
  CHECK_OR_FALSE(im_info_dims[1] == 3);

  return true;
}

bool BoxClipOpLite::InferShapeImpl() const {
  auto* input = param_.Input;
  auto* output = param_.Output;
  output->Resize(input->dims());
  output->set_lod(input->lod());
  return true;
}

bool BoxClipOpLite::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto input = op_desc.Input("Input").front();
  auto im_info = op_desc.Input("ImInfo").front();
  auto output = op_desc.Output("Output").front();
  param_.Input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.ImInfo = scope->FindVar(im_info)->GetMutable<lite::Tensor>();
  param_.Output = scope->FindVar(output)->GetMutable<lite::Tensor>();

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(box_clip, paddle::lite::operators::BoxClipOpLite);
