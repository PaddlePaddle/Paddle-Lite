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

#include <vector>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/apu/bridges/graph.h"
#include "lite/kernels/apu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

int ConcatConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto model = graph->model();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  int neuron_errCode;
  VLOG(3) << "[APU] Converting [" << op_type << "]";

  // Get input and output vars and op attributes
  auto x_names = op_info->Input("X");
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto axis = op_info->GetAttr<int>("axis");
  auto num = x_names.size();

  // Process data layout axis change
  if (axis == 1)
    axis = 3;
  else if (axis == 2)
    axis = 1;
  else if (axis == 3)
    axis = 2;

  // Limitation:
  // All input tensors of NEURON_TENSOR_QUANT8_ASYMM must
  // have the same scale and zeroPoint as the output tensor
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  auto output_scale = op_info->GetOutputScale(out_scale_name, true)[0];

  // Traverse all of input nodes
  std::vector<std::shared_ptr<Node>> input_nodes;
  NeuronOperandType xType;
  for (int i = 0; i < num; i++) {
    auto x_name = x_names[i];
    auto x_scale_name = "X" + paddle::lite::to_string(i) + "_scale";
    auto x = scope->FindMutableTensor(x_name);
    auto x_dims = x->dims();
    std::shared_ptr<Node> x_node = nullptr;

    CHECK(op_info->HasInputScale(x_scale_name, true));
    auto input_scale = op_info->GetInputScale(x_scale_name, true)[0];

    // Add x tensor type
    xType.type = NEURON_TENSOR_QUANT8_ASYMM;
    xType.scale = input_scale;
    xType.zeroPoint = 128;
    xType.dimensionCount = x_dims.size();
    std::vector<uint32_t> dims_x = {(uint32_t)x_dims[0],
                                    (uint32_t)x_dims[2],
                                    (uint32_t)x_dims[3],
                                    (uint32_t)x_dims[1]};
    xType.dimensions = &dims_x[0];
    if (graph->Has(x_name)) {
      VLOG(3) << "Graph has " << x_name;
      if (graph->IsInput(x_name)) {
        VLOG(3) << x_name << "is input and already exist";
        x_name = "transpose_" + x_name;
      }

      if (graph->IsOutput(x_name)) {
        VLOG(3) << x_name << "is input and output node";
        x_name = "transpose_" + x_name;
      }
      x_node = graph->Get(x_name);
    } else {
      // Add input operand
      if (graph->IsInput(x_name)) {
        // Insert transpose for NCHW -> NHWC
        insert_transpose_node(ctx,
                              x_name,
                              "transpose_" + x_name,
                              {(uint32_t)x_dims[0],
                               (uint32_t)x_dims[1],
                               (uint32_t)x_dims[2],
                               (uint32_t)x_dims[3]},
                              dims_x,
                              {0, 2, 3, 1},
                              xType.scale,
                              xType.zeroPoint);

        // Change x_name because we add transpose op
        x_name = "transpose_" + x_name;
        x_node = graph->Get(x_name);
      } else {
        NeuronModel_addOperand(model, &xType);
        x_node = graph->Add(x_name, dims_x);
      }
    }  // End of else
    if (x_node == nullptr) return subgraph::FAILED;
    input_nodes.push_back(x_node);

    VLOG(3) << "input node x: " << x_node->index()
            << ": input_scale: " << input_scale << " x_dims:" << x_dims[0]
            << ":" << x_dims[1] << ":" << x_dims
            << ", inType: " << xType.dimensions[0] << ":" << xType.dimensions[1]
            << ":" << xType.dimensions[2] << ":" << xType.dimensions[3];
  }  // End of for

  if (input_nodes.size() != num) {
    LOG(WARNING) << "Create input operand failed!";
    return subgraph::FAILED;
  }

  // Add axis operand type
  NeuronOperandType int32Type;
  int32Type.type = NEURON_INT32;
  int32Type.dimensionCount = 0;
  std::vector<uint32_t> dims_int32 = {1};

  // Add axis operand
  std::shared_ptr<Node> axis_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // axis
  axis_node = graph->Add(out_name + "_axis", dims_int32);
  VLOG(3) << "axis:" << axis;

  // Add out operand type
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  NeuronOperandType outType;
  outType.type = NEURON_TENSOR_QUANT8_ASYMM;
  outType.scale = output_scale;
  outType.zeroPoint = 128;
  outType.dimensionCount = out_dims.size();
  std::vector<uint32_t> dims_out = {(uint32_t)out_dims[0],
                                    (uint32_t)out_dims[2],
                                    (uint32_t)out_dims[3],
                                    (uint32_t)out_dims[1]};
  outType.dimensions = &dims_out[0];

  // Add out operand
  std::shared_ptr<Node> out_node = nullptr;
  if (graph->Has(out_name)) {
    out_node = graph->Get(out_name);
  } else {
    if (graph->IsOutput(out_name)) {
      NeuronModel_addOperand(model, &outType);
      out_node = graph->Add("transpose_" + out_name, dims_out);
    } else {
      NeuronModel_addOperand(model, &outType);
      out_node = graph->Add(out_name, dims_out);
    }
  }
  VLOG(3) << "out node idx: " << out_node->index()
          << ": output_scle: " << outType.scale
          << ", outType: " << outType.dimensions[0] << ":"
          << outType.dimensions[1] << ":" << outType.dimensions[2] << ":"
          << outType.dimensions[3];

  // Set axis value
  int32_t axis_val[1] = {(int32_t)axis};
  NeuronModel_setOperandValue(
      model, axis_node->index(), axis_val, sizeof(int32_t) * 1);

  std::vector<uint32_t> addInIndex;
  for (auto& node : input_nodes) {
    addInIndex.push_back(node->index());
  }

  addInIndex.push_back(axis_node->index());
  std::vector<uint32_t> addOutIndex = {out_node->index()};
  neuron_errCode = NeuronModel_addOperation(model,
                                            NEURON_CONCATENATION,
                                            addInIndex.size(),
                                            &addInIndex[0],
                                            addOutIndex.size(),
                                            &addOutIndex[0]);

  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Add op fail:" << op_type;
    return subgraph::FAILED;
  }

  if (graph->IsOutput(out_name)) {
    // Insert transpose for NHWC -> NCHW
    insert_transpose_node(ctx,
                          "transpose_" + out_name,
                          out_name,
                          dims_out,
                          {(uint32_t)out_dims[0],
                           (uint32_t)out_dims[1],
                           (uint32_t)out_dims[2],
                           (uint32_t)out_dims[3]},
                          {0, 3, 1, 2},
                          outType.scale,
                          outType.zeroPoint);
    out_node = graph->Get(out_name);
    if (out_node == nullptr) return subgraph::FAILED;
  }

  return SUCCESS;
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(concat,
                         kAPU,
                         paddle::lite::subgraph::apu::ConcatConverter);
