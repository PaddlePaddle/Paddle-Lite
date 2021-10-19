// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/__xpu__squeeze_excitation_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUSqueezeExcitationOp::CheckShape() const {
  CHECK(param_.input)
      << "Input(input) of XPUSqueezeExcitationOp should not be null.";
  CHECK(param_.filter)
      << "Input(weight) of XPUSqueezeExcitationOp should not be null.";
  CHECK(param_.output)
      << "Output(output) of XPUSqueezeExcitationOp should not be null.";
  auto filter_dims = param_.filter_dims;
  auto channel = static_cast<int>(param_.input->dims()[1]);
  CHECK_EQ(channel, filter_dims[1]);

  if (param_.has_branch) {
    const auto in_dims = param_.input->dims();
    const auto branch_dims = param_.branch->dims();
    CHECK_EQ(in_dims.size(), 4UL)
        << "XPUSqueezeExcitationOp intput should be 4-D tensor.";
    CHECK_EQ(branch_dims.size(), 4UL)
        << "XPUSqueezeExcitationOp branch should be 4-D tensor.";
    for (auto i = 0; i < 4; i++) {
      CHECK_EQ(in_dims[i], branch_dims[i]);
    }
  }
  return true;
}

bool XPUSqueezeExcitationOp::InferShapeImpl() const {
  auto input_dim = param_.input->dims();
  param_.output->Resize(input_dim);
  param_.output->set_lod(param_.input->lod());
  return true;
}

bool XPUSqueezeExcitationOp::AttachImpl(const cpp::OpDesc& op_desc,
                                        lite::Scope* scope) {
  CHECK(scope->FindVar(op_desc.Input("Input").front()));
  CHECK(scope->FindVar(op_desc.Input("Filter").front()));
  CHECK(scope->FindVar(op_desc.Output("Output").front()));

  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.filter =
      scope->FindVar(op_desc.Input("Filter").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();

  param_.op_type = op_desc.GetAttr<std::vector<int>>("op_type");
  param_.place_x = op_desc.GetAttr<std::vector<int>>("place_x");
  param_.place_y = op_desc.GetAttr<std::vector<int>>("place_y");
  param_.place_z = op_desc.GetAttr<std::vector<int>>("place_z");
  param_.filter_dims = op_desc.GetAttr<std::vector<int>>("filter_dims");
  CHECK_EQ(param_.filter_dims.size(), 2UL);
  param_.block_lod = op_desc.GetAttr<std::vector<int>>("block_lod");
  param_.act_type = op_desc.GetAttr<std::vector<int>>("act_type");
  CHECK_EQ(param_.act_type.size(), 3UL);
  param_.act_param = op_desc.GetAttr<std::vector<float>>("act_param");
  CHECK_EQ(param_.act_param.size(), 3UL);
  param_.has_branch = op_desc.GetAttr<bool>("has_branch");
  param_.has_bias = op_desc.GetAttr<bool>("has_bias");

  if (op_desc.GetAttr<bool>("has_branch")) {
    CHECK(scope->FindVar(op_desc.Input("Branch").front()));
    param_.branch =
        scope->FindVar(op_desc.Input("Branch").front())->GetMutable<Tensor>();
  }
  if (op_desc.GetAttr<bool>("has_bias")) {
    CHECK(scope->FindVar(op_desc.Input("Bias").front()));
    param_.bias =
        scope->FindVar(op_desc.Input("Bias").front())->GetMutable<Tensor>();
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__squeeze_excitation_block,
                 paddle::lite::operators::XPUSqueezeExcitationOp);
