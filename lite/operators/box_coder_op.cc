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

#include "lite/operators/box_coder_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool BoxCoderOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.prior_box);
  CHECK_OR_FALSE(param_.target_box);
  CHECK_OR_FALSE(param_.proposals);

  auto prior_box_dims = param_.prior_box->dims();
  CHECK_OR_FALSE(prior_box_dims.size() == 2);
  CHECK_OR_FALSE(prior_box_dims[1] == 4);
  if (param_.prior_box_var != nullptr) {
    auto box_var_dim = param_.prior_box_var->dims();
    CHECK_OR_FALSE(box_var_dim.size() == 2);
    CHECK_OR_FALSE(box_var_dim == prior_box_dims);
  }
  return true;
}

bool BoxCoderOpLite::InferShapeImpl() const {
  auto prior_box_dims = param_.prior_box->dims();
  auto target_box_dims = param_.target_box->dims();
  std::string code_type = param_.code_type;
  int axis = param_.axis;
  CHECK_OR_FALSE(code_type == "encode_center_size" ||
                 code_type == "decode_center_size");

  if (code_type == "encode_center_size") {
    CHECK_OR_FALSE(target_box_dims.size() == 2);
    CHECK_OR_FALSE(target_box_dims[1] == 4);
    param_.proposals->Resize({target_box_dims[0], prior_box_dims[0], 4});
  } else if (code_type == "decode_center_size") {
    CHECK_OR_FALSE(target_box_dims.size() == 3);
    CHECK_OR_FALSE(axis == 0 || axis == 1);
    if (axis == 0) {
      CHECK_OR_FALSE(target_box_dims[1] == prior_box_dims[0]);
    } else if (axis == 1) {
      CHECK_OR_FALSE(target_box_dims[0] == prior_box_dims[0]);
    }
    CHECK_OR_FALSE(target_box_dims[2] == prior_box_dims[1]);
    param_.proposals->Resize(target_box_dims);
  }
  if (code_type == "decode_center_size" && axis == 1) {
    param_.proposals->set_lod(param_.prior_box->lod());
  } else {
    param_.proposals->set_lod(param_.target_box->lod());
  }
  return true;
}

bool BoxCoderOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto Prior_box_name = opdesc.Input("PriorBox").front();
  auto Target_box_name = opdesc.Input("TargetBox").front();
  auto Output_box_name = opdesc.Output("OutputBox").front();
  param_.prior_box = GetVar<lite::Tensor>(scope, Prior_box_name);
  param_.target_box = GetVar<lite::Tensor>(scope, Target_box_name);
  param_.proposals = GetMutableVar<lite::Tensor>(scope, Output_box_name);
  // optional params
  std::vector<std::string> input_arg_names = opdesc.InputArgumentNames();
  if (std::find(input_arg_names.begin(),
                input_arg_names.end(),
                "PriorBoxVar") != input_arg_names.end()) {
    auto box_var_arguments = opdesc.Input("PriorBoxVar");
    if (box_var_arguments.size() > 0) {
      auto* box_var = scope->FindVar(box_var_arguments.front());
      if (box_var != nullptr) {
        param_.prior_box_var = box_var->GetMutable<Tensor>();
      }
    }
  }

  param_.code_type = opdesc.GetAttr<std::string>("code_type");
  param_.box_normalized = opdesc.GetAttr<bool>("box_normalized");
  if (opdesc.HasAttr("axis")) {
    param_.axis = opdesc.GetAttr<int>("axis");
  }

  if (opdesc.HasAttr("variance")) {
    param_.variance = opdesc.GetAttr<std::vector<float>>("variance");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(box_coder, paddle::lite::operators::BoxCoderOpLite);
