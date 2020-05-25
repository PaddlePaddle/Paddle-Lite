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

#include "lite/operators/decode_bboxes_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool DecodeBboxesOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.loc_data);
  CHECK_OR_FALSE(param_.prior_data);
  CHECK_OR_FALSE(param_.bbox_data);

  CHECK_EQ(param_.loc_data->dims().size(), 2);
  CHECK_EQ(param_.prior_data->dims().size(), 3);
  return true;
}

bool DecodeBboxesOpLite::InferShapeImpl() const {
  param_.bbox_data->Resize(param_.loc_data->dims());
  return true;
}

bool DecodeBboxesOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                    lite::Scope* scope) {
  auto Loc_name = opdesc.Input("Loc").front();
  auto Prior_name = opdesc.Input("Prior").front();
  auto Bbox_name = opdesc.Output("Bbox").front();
  param_.loc_data = GetVar<lite::Tensor>(scope, Loc_name);
  param_.prior_data = GetVar<lite::Tensor>(scope, Prior_name);
  param_.bbox_data = GetMutableVar<lite::Tensor>(scope, Bbox_name);

  param_.batch_num = opdesc.GetAttr<int>("batch_num");
  param_.num_priors = opdesc.GetAttr<int>("num_priors");
  param_.num_loc_classes = opdesc.GetAttr<int>("num_loc_classes");
  param_.share_location = opdesc.GetAttr<bool>("share_location");
  param_.variance_encoded_in_target =
      opdesc.GetAttr<bool>("variance_encoded_in_target");
  param_.code_type = opdesc.GetAttr<std::string>("code_type");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(decode_bboxes, paddle::lite::operators::DecodeBboxesOpLite);
