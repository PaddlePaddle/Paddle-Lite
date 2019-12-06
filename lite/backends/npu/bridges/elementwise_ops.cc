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

#include "lite/backends/npu/bridges/registry.h"
#include "lite/backends/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridges {

std::vector<int64_t> CvtYShape(const Tensor& x, Tensor* y, int axis) {
  auto x_dims = x.dims();
  CHECK_EQ(x_dims.size(), 4UL) << "[NPU] Only support 4-dimension x";
  auto y_dims = y->dims();
  CHECK_GE(x_dims.size(), y_dims.size());

  if (axis < 0) {
    axis += x_dims.size();
  }

  std::vector<int64_t> y_new_shape(y_dims.Vectorize());
  if (y_new_shape.size() == 4UL) {
    return y_new_shape;
  }
  for (int i = 0; i < axis; i++) {
    y_new_shape.insert(y_new_shape.begin(), 1);
  }
  while (y_new_shape.size() < 4) {
    y_new_shape.push_back(1);
  }
  CHECK_EQ(y_new_shape.size(), 4UL);
  return y_new_shape;
}

int ElementwiseConverter(cvt_ctx_type* ctx, lite::OpLite* op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  auto out_var_name = op_info->Output("Out").front();
  auto axis = op_info->GetAttr<int>("axis");

  std::shared_ptr<ge::Operator> elementwise_node = nullptr;
  CHECK(!ctx->HasNode(x_var_name));
  std::shared_ptr<ge::Operator> x_node = ctx->GetNode(x_var_name);
  std::shared_ptr<ge::Operator> y_node = nullptr;
  if (ctx->HasNode(y_var_name)) {
    y_node = ctx->GetNode(y_var_name);
  } else {
    auto y_const_node = ctx->AddNode<ge::op::Const>(y_var_name);
    auto x = scope->FindTensor(x_var_name);
    auto y = scope->FindMutableTensor(y_var_name);
    auto y_new_shape = CvtYShape(*x, y, axis);
    y_const_node->set_attr_value(CvtTensor(y, y_new_shape));
    y_node = y_const_node;
  }

  if (op_type == "elementwise_add" ||
      op_type == "fusion_elementwise_add_activation") {
    auto elt_node = ctx->AddNode<ge::op::Add>(out_var_name);
    elt_node->set_input_x1(*x_node);
    elt_node->set_input_x2(*y_node);
    elementwise_node = elt_node;
  } else if (op_type == "elementwise_sub") {
    auto elt_node = ctx->AddNode<ge::op::Sub>(out_var_name);
    elt_node->set_input_x1(*x_node);
    elt_node->set_input_x2(*y_node);
    elementwise_node = elt_node;
  } else if (op_type == "elementwise_mul") {
    auto elt_node = ctx->AddNode<ge::op::Mul>(out_var_name);
    elt_node->set_input_x(*x_node);
    elt_node->set_input_y(*y_node);
    elementwise_node = elt_node;
  } else if (op_type == "elementwise_div") {
    auto elt_node = ctx->AddNode<ge::op::RealDiv>(out_var_name);
    elt_node->set_input_x1(*x_node);
    elt_node->set_input_x2(*y_node);
    elementwise_node = elt_node;
  } else {
    LOG(WARNING) << "[NPU] Unsupported op type: " << op_type;
    return FAILED;
  }

  if (op_type == "fusion_elementwise_add_activation") {
    auto act_type = op_info->GetAttr<std::string>("act_type");
    auto act_node = ctx->AddNode<ge::op::Activation>(out_var_name);
    act_node->set_input_x(*elementwise_node);
    // TODO(hong19860320) set the coef value for act Ops, such as leaky_relu,
    // clipped_relu etc.
    act_node->set_attr_mode(CvtActMode(act_type));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace bridges
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(elementwise_add,
                    paddle::lite::npu::bridges::ElementwiseConverter);
REGISTER_NPU_BRIDGE(fusion_elementwise_add_activation,
                    paddle::lite::npu::bridges::ElementwiseConverter);
REGISTER_NPU_BRIDGE(elementwise_sub,
                    paddle::lite::npu::bridges::ElementwiseConverter);
REGISTER_NPU_BRIDGE(elementwise_mul,
                    paddle::lite::npu::bridges::ElementwiseConverter);
REGISTER_NPU_BRIDGE(elementwise_div,
                    paddle::lite::npu::bridges::ElementwiseConverter);
