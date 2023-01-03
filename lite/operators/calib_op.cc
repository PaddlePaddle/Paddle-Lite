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

#include "lite/operators/calib_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CalibOpLite::CheckShape() const {
  CHECK(param_.input);
  CHECK(param_.output);
  return true;
}

bool CalibOpLite::InferShapeImpl() const {
  param_.output->Resize(param_.input->dims());
  param_.output->set_lod(param_.input->lod());
  return true;
}

#ifdef LITE_ON_FLATBUFFERS_DESC_VIEW
bool CalibOpLite::AttachImpl(const cpp::OpDescWrite &opdesc,
                             lite::Scope *scope) {
  param_.input = scope->FindTensor(opdesc.Input("Input").front());
  param_.output = scope->FindMutableTensor(opdesc.Output("Out").front());
  if (opdesc.HasAttr("scale")) {
    param_.scale = opdesc.GetAttr<float>("scale");
  }
  return true;
}
#endif

bool CalibOpLite::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.input = scope->FindTensor(opdesc.Input("Input").front());
  param_.output = scope->FindMutableTensor(opdesc.Output("Out").front());
  if (opdesc.HasAttr("scale")) {
    param_.scale = opdesc.GetAttr<float>("scale");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(calib, paddle::lite::operators::CalibOpLite);
