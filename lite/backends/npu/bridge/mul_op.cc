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

#include "lite/operators/mul_op.h"
#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "ai_ddk_lib/include/graph/operator.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/utils.h"
#include "lite/backends/npu/npu_helper.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

// Note: inputs_map the var_name contains only the data, the weight should be
// handle in this converter
node_map_type MulConverter(const std::shared_ptr<lite::OpLite> mul_op,
                           const node_map_type& inputs_map) {
  LOG(INFO) << "converting mul...";
  lite::Scope* scope = mul_op->scope();
  const lite::OpInfo* op_info = mul_op->op_info();
  auto output_node = std::make_shared<ge::op::MatMul>(UniqueName("mul"));

  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  int x_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  int y_num_col_dims = op_info->GetAttr<int>("y_num_col_dims");
  auto* xtensor = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto* ytensor = scope->FindVar(y_var_name)->GetMutable<lite::Tensor>();

  int m = xtensor->dims().Slice(0, x_num_col_dims).production();
  int x_w = xtensor->dims()
                .Slice(x_num_col_dims, xtensor->dims().size())
                .production();
  int y_h = ytensor->dims().Slice(0, y_num_col_dims).production();
  int n = ytensor->dims()
              .Slice(y_num_col_dims, ytensor->dims().size())
              .production();
  CHECK_EQ(x_w, y_h) << "x_w must be equal with y_h";
  int k = x_w;
  LOG(INFO) << "m:" << m << ",n:" << n << ",k:" << k;
  LOG(INFO) << "x_var_name:" << x_var_name
            << ", is data: " << inputs_map.count(x_var_name);
  LOG(INFO) << "y_var_name:" << y_var_name
            << ", is data: " << inputs_map.count(y_var_name);
  CHECK(inputs_map.count(x_var_name))
      << "[NPU] MatMul only support X is data, Y is const yet";
  if (inputs_map.count(x_var_name)) {
    auto xsrc = inputs_map.at(x_var_name);
    auto reshapex = std::make_shared<ge::op::Reshape>(x_var_name + "_reshape");
    reshapex->set_input_tensor(*xsrc);
    reshapex->set_attr_shape({m, k});
    reshapex->set_attr_axis(0);
    OpList::Global().add(xsrc);
    OpList::Global().add(reshapex);
    output_node->set_input_x(*reshapex);
  } else {
    auto constx = std::make_shared<ge::op::Const>(x_var_name);
    ge::TensorDesc desc(ge::Shape({m, k}), ge::FORMAT_NCHW, ge::DT_FLOAT);
    auto size = desc.GetShape().GetShapeSize();
    CHECK_EQ(size, xtensor->dims().production());
    ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
    ptensor->SetTensorDesc(desc);
    auto* pdata = reinterpret_cast<uint8_t*>(xtensor->mutable_data<float>());
    ptensor->SetData(pdata, size * sizeof(float));
    constx->set_attr_value(ptensor);
    OpList::Global().add(constx);
    output_node->set_input_x(*constx);
  }

  if (inputs_map.count(y_var_name)) {
    auto ysrc = inputs_map.at(y_var_name);
    auto reshapey = std::make_shared<ge::op::Reshape>(y_var_name + "_reshape");
    reshapey->set_input_tensor(*ysrc);
    reshapey->set_attr_shape({k, n});
    reshapey->set_attr_axis(0);
    OpList::Global().add(ysrc);
    OpList::Global().add(reshapey);
    output_node->set_input_w(*reshapey);
  } else {
    auto consty = std::make_shared<ge::op::Const>(y_var_name);
    ge::TensorDesc desc(ge::Shape({k, n}), ge::FORMAT_NCHW, ge::DT_FLOAT);
    auto size = desc.GetShape().GetShapeSize();
    CHECK_EQ(size, ytensor->dims().production());
    ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
    ptensor->SetTensorDesc(desc);
    auto* pdata = reinterpret_cast<uint8_t*>(ytensor->mutable_data<float>());
    ptensor->SetData(pdata, size * sizeof(float));
    consty->set_attr_value(ptensor);
    OpList::Global().add(consty);
    output_node->set_input_w(*consty);
  }

  OpList::Global().add(output_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = output_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(mul, paddle::lite::npu::bridge::MulConverter);
