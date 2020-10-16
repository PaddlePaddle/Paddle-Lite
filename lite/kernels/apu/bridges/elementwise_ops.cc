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

int ElementwiseConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto model = graph->model();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  int neuron_errCode;
  VLOG(3) << "[APU] Converting [" + op_type + "]";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto x = scope->FindTensor(x_name);
  auto x_dims = x->dims();

  auto y_name = op_info->Input("Y").front();
  auto y_scale_name = "Y0_scale";
  auto y = scope->FindTensor(y_name);
  auto y_dims = y->dims();

  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto out = scope->FindTensor(out_name);
  auto out_dims = out->dims();

  auto axis = op_info->GetAttr<int>("axis");
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }

  auto x_shape = x_dims.Vectorize();
  auto y_shape = y_dims.Vectorize();

  // Two dimensions are compatible when:
  // 1. they are equal, or
  // 2. one of them is 1
  for (int i = axis; i < x_shape.size(); i++) {
    if (x_dims[i] != y_dims[i - axis]) {
      // Input 1 compatible dimensions as input0
      if (y_dims[i - axis] != 1) {
        LOG(WARNING) << i << ":" << axis << ":" << y_dims[i - axis];
        return FAILED;
      }
    }
  }  // End of for

  int32_t fuse_val[1] = {NEURON_FUSED_NONE};
  // Act node
  if (op_type == "fusion_elementwise_add_activation" ||
      op_type == "fusion_elementwise_sub_activation" ||
      op_type == "fusion_elementwise_mul_activation" ||
      op_type == "fusion_elementwise_div_activation") {
    auto act_type = op_info->GetAttr<std::string>("act_type");

    if (act_type == "relu") {
      fuse_val[0] = NEURON_FUSED_RELU;
    } else if (act_type == "relu1") {
      fuse_val[0] = NEURON_FUSED_RELU1;
    } else if (act_type == "relu6") {
      fuse_val[0] = NEURON_FUSED_RELU6;
    } else if (!act_type.empty()) {
      fuse_val[0] = NEURON_FUSED_NONE;
      LOG(WARNING) << "Support act_type: " << act_type;
      return FAILED;
    }
  }  // End of if
  VLOG(3) << "x_name" << x_name;

  CHECK(op_info->HasInputScale(x_scale_name, true));
  auto x_scale = op_info->GetInputScale(x_scale_name, true)[0];
  CHECK(op_info->HasInputScale(y_scale_name, true));
  auto y_scale = op_info->GetInputScale(y_scale_name, true)[0];
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  auto out_scale = op_info->GetOutputScale(out_scale_name, true)[0];

  // Add x tensor type
  NeuronOperandType xType;
  xType.type = NEURON_TENSOR_QUANT8_ASYMM;
  xType.scale = x_scale;
  xType.zeroPoint = 128;
  xType.dimensionCount = x_dims.size();
  std::vector<uint32_t> dims_x = {(uint32_t)x_dims[0],
                                  (uint32_t)x_dims[2],
                                  (uint32_t)x_dims[3],
                                  (uint32_t)x_dims[1]};
  xType.dimensions = &dims_x[0];

  std::shared_ptr<Node> x_node = nullptr;
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
    if (graph->IsInput(x_name)) {
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

      // Change x name after insert transpose op for x data relayout
      x_name = "transpose_" + x_name;
      x_node = graph->Get(x_name);
    } else {
      NeuronModel_addOperand(model, &xType);
      x_node = graph->Add(x_name, dims_x);
    }
  }  // End of else
  VLOG(3) << "x node idx: " << x_node->index() << "x_dims: " << x_dims
          << ": x_scale: " << x_scale << ", xType: " << xType.dimensions[0]
          << ":" << xType.dimensions[1] << ":" << xType.dimensions[2] << ":"
          << xType.dimensions[3];

  // Add y tensor type
  NeuronOperandType yType;
  yType.type = NEURON_TENSOR_QUANT8_ASYMM;
  yType.scale = y_scale;
  yType.zeroPoint = 128;
  yType.dimensionCount = y_dims.size();
  std::vector<uint32_t> dims_y = {(uint32_t)y_dims[0],
                                  (uint32_t)y_dims[2],
                                  (uint32_t)y_dims[3],
                                  (uint32_t)y_dims[1]};
  yType.dimensions = &dims_y[0];

  std::shared_ptr<Node> y_node = nullptr;
  if (graph->Has(y_name)) {
    VLOG(3) << "Graph has " << y_name;
    y_node = graph->Get(y_name);
  } else {
    if (graph->IsInput(y_name)) {
      insert_transpose_node(ctx,
                            y_name,
                            "transpose_" + y_name,
                            {(uint32_t)y_dims[0],
                             (uint32_t)y_dims[1],
                             (uint32_t)y_dims[2],
                             (uint32_t)y_dims[3]},
                            dims_y,
                            {0, 2, 3, 1},
                            yType.scale,
                            yType.zeroPoint);

      y_name = "transpose_" + y_name;
      y_node = graph->Get(y_name);
    } else {
      NeuronModel_addOperand(model, &yType);
      y_node = graph->Add(y_name, dims_y);
    }
  }
  VLOG(3) << "y node idx: " << y_node->index() << "y_dims: " << y_dims
          << ": y_scale: " << y_scale << ", yType: " << yType.dimensions[0]
          << ":" << yType.dimensions[1] << ":" << yType.dimensions[2] << ":"
          << yType.dimensions[3];

  // Add fuse operand type
  NeuronOperandType int32Type;
  int32Type.type = NEURON_INT32;
  int32Type.dimensionCount = 0;
  std::vector<uint32_t> dims_int32 = {1};

  // Add fuse operand
  std::shared_ptr<Node> fuse_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 2: fuse
  fuse_node = graph->Add(out_name + "_fuse", dims_int32);

  // Add out tensor type
  NeuronOperandType outType;
  outType.type = NEURON_TENSOR_QUANT8_ASYMM;
  outType.scale = out_scale;
  outType.zeroPoint = 128;
  outType.dimensionCount = out_dims.size();
  std::vector<uint32_t> dims_out = {(uint32_t)out_dims[0],
                                    (uint32_t)out_dims[2],
                                    (uint32_t)out_dims[3],
                                    (uint32_t)out_dims[1]};
  outType.dimensions = &dims_out[0];

  std::shared_ptr<Node> out_node = nullptr;
  if (graph->Has(out_name)) {
    VLOG(3) << "Graph has " << out_name;
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
  VLOG(3) << "out node idx: " << out_node->index() << "out_dims: " << out_dims
          << ": out_scale: " << out_scale
          << ", outType: " << outType.dimensions[0] << ":"
          << outType.dimensions[1] << ":" << outType.dimensions[2] << ":"
          << outType.dimensions[3];

  // Set fuse value
  NeuronModel_setOperandValue(
      model, fuse_node->index(), fuse_val, sizeof(int32_t) * 1);

  std::vector<uint32_t> addInIndex = {
      x_node->index(),      // 0: A tensor
      y_node->index(),      // 1: A tensor of the same OperandCode,
                            //    and compatible dimensions as input 0
      fuse_node->index()};  // 2: fuse

  std::vector<uint32_t> addOutIndex = {out_node->index()};
  if (op_type == "elementwise_add" ||
      op_type == "fusion_elementwise_add_activation") {
    neuron_errCode = NeuronModel_addOperation(model,
                                              NEURON_ADD,
                                              addInIndex.size(),
                                              &addInIndex[0],
                                              addOutIndex.size(),
                                              &addOutIndex[0]);
  } else {
    LOG(WARNING) << "[APU] Unsupported op type: " << op_type;
    return FAILED;
  }

  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "ADD op fail:" << op_type;
    return FAILED;
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
    if (out_node == nullptr) return FAILED;
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(elementwise_add,
                         kAPU,
                         paddle::lite::subgraph::apu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_mul,
                         kAPU,
                         paddle::lite::subgraph::apu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(fusion_elementwise_add_activation,
                         kAPU,
                         paddle::lite::subgraph::apu::ElementwiseConverter);
