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

#include "lite/operators/fc_op.h"
#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "ai_ddk_lib/include/graph/operator.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/utils.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type FCConverter(const std::shared_ptr<lite::OpLite> fc_op,
                          const node_map_type& inputs_map) {
  LOG(INFO) << "Converting fc...";
  lite::Scope* scope = fc_op->scope();
  const lite::OpInfo* op_info = fc_op->op_info();
  auto output_node = std::make_shared<ge::op::MatMul>(UniqueName("fc"));

  auto x_var_name = op_info->Input("Input").front();
  auto w_var_name = op_info->Input("W").front();

  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  auto* xtensor = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto* wtensor = scope->FindVar(w_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = xtensor->dims();
  auto w_dims = wtensor->dims();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);

  int m = x_dims.Slice(0, in_num_col_dims).production();
  int k = x_dims.Slice(in_num_col_dims, x_dims.size()).production();
  int n = w_dims[1];

  CHECK(inputs_map.count(x_var_name));
  CHECK(!inputs_map.count(w_var_name));

  LOG(INFO) << "m:" << m << ",n:" << n << ",k:" << k;
  LOG(INFO) << "x_var_name:" << x_var_name
            << ", is data: " << inputs_map.count(x_var_name);
  LOG(INFO) << "w_var_name:" << w_var_name
            << ", is data: " << inputs_map.count(w_var_name);

  auto xsrc = inputs_map.at(x_var_name);
  auto reshapex = std::make_shared<ge::op::Reshape>(x_var_name + "_reshape");
  reshapex->set_input_tensor(*xsrc);
  reshapex->set_attr_shape({m, k});
  reshapex->set_attr_axis(0);
  OpList::Global().add(xsrc);
  OpList::Global().add(reshapex);
  output_node->set_input_x(*reshapex);

  auto wconst = std::make_shared<ge::op::Const>(w_var_name);
  ge::TensorDesc wdesc(ge::Shape({k, n}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto size = wdesc.GetShape().GetShapeSize();
  CHECK_EQ(size, w_dims.production());
  ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
  ptensor->SetTensorDesc(wdesc);
  auto* pdata = reinterpret_cast<uint8_t*>(wtensor->mutable_data<float>());
  ptensor->SetData(pdata, size * sizeof(float));
  wconst->set_attr_value(ptensor);
  OpList::Global().add(wconst);
  output_node->set_input_w(*wconst);

  if (HasInputArg(op_info, scope, "Bias")) {
    auto b_var_name = op_info->Input("Bias").front();
    auto* btensor = scope->FindVar(b_var_name)->GetMutable<lite::Tensor>();

    LOG(INFO) << "b_var_name:" << b_var_name
              << ", is data: " << inputs_map.count(b_var_name);
    CHECK(!inputs_map.count(b_var_name));
    CHECK_EQ(btensor->numel(), n);

    auto bconst = std::make_shared<ge::op::Const>(b_var_name);
    ge::TensorDesc bdesc(
        ge::Shape({1, n, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
    auto size = bdesc.GetShape().GetShapeSize();
    CHECK_EQ(size, n);
    ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
    ptensor->SetTensorDesc(bdesc);
    auto* pdata = reinterpret_cast<uint8_t*>(btensor->mutable_data<float>());
    ptensor->SetData(pdata, size * sizeof(float));
    bconst->set_attr_value(ptensor);
    OpList::Global().add(bconst);
    output_node->set_input_bias(*bconst);
    output_node->set_attr_has_bias(ge::AttrValue::BOOL{true});
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

REGISTER_NPU_BRIDGE(fc, paddle::lite::npu::bridge::FCConverter);
