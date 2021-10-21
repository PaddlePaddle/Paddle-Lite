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

#include "lite/operators/__xpu__block_fuse_op.h"
#include <memory>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUBlockFuseOp::CheckShape() const {
  CHECK(param_.input) << "Input(input) of XPUBlockFuseOp should not be null.";
  CHECK(param_.output)
      << "Output(output) of XPUBlockFuseOp should not be null.";
  return true;
}

bool XPUBlockFuseOp::InferShapeImpl() const {
  param_.output_max->Resize({4});
  param_.output->set_lod(param_.input->lod());
  return true;
}

bool XPUBlockFuseOp::AttachImpl(const cpp::OpDesc& op_desc,
                                lite::Scope* scope) {
  CHECK(scope->FindVar(op_desc.Input("Input").front()));
  CHECK(scope->FindVar(op_desc.Output("Output").front()));
  CHECK(scope->FindVar(op_desc.Output("OutputMax").front()));

  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();
  param_.filter =
      scope->FindVar(op_desc.Input("Filter").front())->GetMutable<Tensor>();
  param_.output_max =
      scope->FindVar(op_desc.Output("OutputMax").front())->GetMutable<Tensor>();

  param_.op_type = op_desc.GetAttr<std::vector<int>>("op_type");
  param_.place_x = op_desc.GetAttr<std::vector<int>>("place_x");
  param_.place_y = op_desc.GetAttr<std::vector<int>>("place_y");
  param_.place_z = op_desc.GetAttr<std::vector<int>>("place_z");
  param_.filter_dims = op_desc.GetAttr<std::vector<int>>("filter_dims");
  param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
  auto paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  param_.paddings = std::make_shared<std::vector<int>>(paddings);
  auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
  param_.dilations = std::make_shared<std::vector<int>>(dilations);
  param_.groups = op_desc.GetAttr<std::vector<int>>("groups");
  param_.act_type = op_desc.GetAttr<std::vector<int>>("act_type");
  param_.act_param = op_desc.GetAttr<std::vector<float>>("act_param");
  param_.block_lod = op_desc.GetAttr<std::vector<int>>("block_lod");
  param_.conv_bias = op_desc.GetAttr<std::vector<int>>("conv_bias");
  param_.has_bias = op_desc.GetAttr<bool>("has_bias");

  // optional params
  if (op_desc.HasAttr("has_input_max") &&
      op_desc.GetAttr<bool>("has_input_max")) {
    CHECK(scope->FindVar(op_desc.Input("InputMax").front()));
    param_.input_max =
        scope->FindVar(op_desc.Input("InputMax").front())->GetMutable<Tensor>();
  }
  if (op_desc.GetAttr<bool>("has_bias")) {
    CHECK(scope->FindVar(op_desc.Input("Bias").front()));
    param_.bias =
        scope->FindVar(op_desc.Input("Bias").front())->GetMutable<Tensor>();
  }

  // shape check
  int op_num = param_.op_type.size();
  CHECK_EQ(op_num, param_.place_x.size());
  CHECK_EQ(op_num, param_.place_y.size());
  CHECK_EQ(op_num, param_.place_z.size());
  int f_n = 0, s_n = 0, p_n = 0, d_n = 0, g_n = 0, act_n = 0, act_param_n = 0,
      bias_n = 0;
  for (size_t i = 0; i < op_num; i++) {
    if (param_.op_type[i] == 0) {
      f_n += 4;
      s_n += 2;
      p_n += 4;
      d_n += 2;
      g_n += 1;
      act_n += 1;
      act_param_n += 1;
      bias_n += 1;
    } else if (param_.op_type[i] <= 3) {
      f_n += 2;
      s_n += 2;
      p_n += 4;
    } else if (param_.op_type[i] == 4) {
      f_n += 2;
      act_n += 3;
      act_param_n += 3;
      bias_n += 1;
    } else if (param_.op_type[i] == 10) {
      act_n += 1;
      act_param_n += 1;
    }
  }
  CHECK_EQ(f_n, param_.filter_dims.size());
  CHECK_EQ(s_n, param_.strides.size());
  CHECK_EQ(p_n, paddings.size());
  CHECK_EQ(d_n, dilations.size());
  CHECK_EQ(g_n, param_.groups.size());
  CHECK_EQ(act_n, param_.act_type.size());
  CHECK_EQ(act_param_n, param_.act_param.size());
  CHECK_EQ(bias_n, param_.conv_bias.size());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__block_fuse_op, paddle::lite::operators::XPUBlockFuseOp);
