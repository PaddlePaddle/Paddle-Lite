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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/apu/bridges/graph.h"
#include "lite/kernels/apu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

int SoftmaxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto model = graph->model();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[APU] Converting [" + op_type + "]";

  CHECK(op_info->HasAttr("enable_int8") &&
        op_info->GetAttr<bool>("enable_int8"));

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_scale_name = "X0_scale";
  auto x_dims = x->dims();
  CHECK_GE(x_dims.size(), 2UL);
  auto x_rank = x_dims.size();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";

  // Check output shape
  auto axis = op_info->GetAttr<int>("axis");
  if (axis < 0) {
    axis += x_rank;
  }

  CHECK(op_info->HasInputScale(x_scale_name, true));
  auto input_scale = op_info->GetInputScale(x_scale_name, true)[0];
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  auto out_scale = op_info->GetOutputScale(out_scale_name, true)[0];

  // Check output scale
  NeuronOperandType xType;
  xType.type = NEURON_TENSOR_QUANT8_ASYMM;
  xType.scale = input_scale;
  xType.zeroPoint = 128;
  xType.dimensionCount = x_dims.size();
  std::vector<uint32_t> dims_x;
  for (int i = 0; i < x_dims.size(); i++) dims_x.push_back(x_dims[i]);
  xType.dimensions = &dims_x[0];
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
    VLOG(3) << "Graph has " << x_name << ",index: " << x_node->index();
  } else {
    NeuronModel_addOperand(model, &xType);  // Operand 0: input
    x_node = graph->Add(x_name, dims_x);
  }
  VLOG(3) << "input_scale size: " << input_scale
          << " ,x_dims size: " << x_dims.size() << " ,x_dims: " << x_dims;

  // Add beta operand
  std::vector<uint32_t> dims_int32 = {0};
  NeuronOperandType betaType;
  betaType.type = NEURON_FLOAT32;
  betaType.dimensionCount = 0;
  NeuronModel_addOperand(model, &betaType);  // Operand 1: beta
  std::shared_ptr<Node> beta_node = nullptr;
  beta_node = graph->Add(x_name + "_beta", dims_int32);

  // Add axis operand
  NeuronOperandType axisType;
  axisType.type = NEURON_INT32;
  axisType.dimensionCount = 0;
  NeuronModel_addOperand(model, &axisType);  // Operand 2: axis
  std::shared_ptr<Node> axis_node = nullptr;
  axis_node = graph->Add(x_name + "_axis", dims_int32);

  // Add out operand
  NeuronOperandType outType;
  outType.type = NEURON_TENSOR_QUANT8_ASYMM;
  outType.scale = out_scale;
  outType.zeroPoint = 128;
  outType.dimensionCount = x_dims.size();
  outType.dimensions = &dims_x[0];
  NeuronModel_addOperand(model, &outType);  // Operand 3: output
  std::shared_ptr<Node> out_node = nullptr;
  out_node = graph->Add(out_name, dims_x);
  VLOG(3) << "out_scale: " << out_scale;

  float beta_val[] = {1.0f};
  NeuronModel_setOperandValue(
      model, beta_node->index(), beta_val, sizeof(float) * 1);

  int32_t axis_val[1];
  axis_val[0] = axis;
  NeuronModel_setOperandValue(
      model, axis_node->index(), axis_val, sizeof(int32_t) * 1);
  std::vector<uint32_t> addInIndex = {x_node->index(),      // 0: input
                                      beta_node->index(),   // 1: beta
                                      axis_node->index()};  // 2: axis
  std::vector<uint32_t> addOutIndex = {out_node->index()};
  int neuron_errCode = NeuronModel_addOperation(model,
                                                NEURON_SOFTMAX,
                                                addInIndex.size(),
                                                &addInIndex[0],
                                                addOutIndex.size(),
                                                &addOutIndex[0]);
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Add op fail:" << op_type;
    return FAILED;
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(softmax,
                         kAPU,
                         paddle::lite::subgraph::apu::SoftmaxConverter);
