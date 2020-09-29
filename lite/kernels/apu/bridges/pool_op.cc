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

#include "lite/operators/pool_op.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/apu/bridges/graph.h"
#include "lite/kernels/apu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

int PoolConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto model = graph->model();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[APU] Converting [" + op_type + "] ";

  CHECK(op_info->HasAttr("enable_int8") &&
        op_info->GetAttr<bool>("enable_int8"));

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");

  // Check pool mode
  if ((pooling_type == "max") || (pooling_type == "avg")) {
  } else {
    LOG(WARNING) << "[APU] Unsupported pooling type: " << pooling_type;
    return FAILED;
  }

  // Check padding mode
  int pad_mode = 0;
  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  if (padding_algorithm == "SAME") {
    pad_mode = 6;
  } else if (padding_algorithm == "VALID") {
    pad_mode = 5;
  }

  // Check paddings and strides
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < 2L; ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "[APU] Paddings size should be the same or twice as the inputs size.";

  bool adaptive = false;
  if (op_info->HasAttr("adaptive")) {
    adaptive = op_info->GetAttr<bool>("adaptive");
  }
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  lite::operators::UpdatePadding(&paddings,
                                 global_pooling,
                                 adaptive,
                                 padding_algorithm,
                                 x->dims(),
                                 strides,
                                 ksize);

  // Add x tensor type
  CHECK(op_info->HasInputScale(x_scale_name, true));
  auto x_scale = op_info->GetInputScale(x_scale_name, true)[0];
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  auto out_scale = op_info->GetOutputScale(out_scale_name, true)[0];

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
    x_node = graph->Get(x_name);
  } else {
    NeuronModel_addOperand(model, &xType);  // Operand 0: x
    x_node = graph->Add(x_name, dims_x);
  }
  VLOG(3) << "x_scale: " << x_scale << ", xType: " << xType.dimensions[0] << ":"
          << xType.dimensions[1] << ":" << xType.dimensions[2] << ":"
          << xType.dimensions[3];

  VLOG(3) << "ksize:" << ksize[0] << ":" << ksize[1];

  NeuronOperandType int32Type;
  int32Type.type = NEURON_INT32;
  int32Type.dimensionCount = 0;
  std::vector<uint32_t> dims_int32 = {0};

  std::shared_ptr<Node> paddingL_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 1: padding left
  paddingL_node = graph->Add(x_name + "_padding_left", dims_int32);

  std::shared_ptr<Node> paddingR_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 2: padding right
  paddingR_node = graph->Add(x_name + "_padding_right", dims_int32);

  std::shared_ptr<Node> paddingT_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 3: padding top
  paddingT_node = graph->Add(x_name + "_padding_top", dims_int32);

  std::shared_ptr<Node> paddingB_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 4: padding bottom
  paddingB_node = graph->Add(x_name + "_padding_bottom", dims_int32);

  std::shared_ptr<Node> strideW_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 5: stride width
  strideW_node = graph->Add(x_name + "_stride_width", dims_int32);

  std::shared_ptr<Node> strideH_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 6: stride height
  strideH_node = graph->Add(x_name + "_stride_height", dims_int32);

  std::shared_ptr<Node> filterW_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 7: filter width
  filterW_node = graph->Add(x_name + "_filter_width", dims_int32);

  std::shared_ptr<Node> filterH_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 8: filter height
  filterH_node = graph->Add(x_name + "_filter_height", dims_int32);

  std::shared_ptr<Node> fuse_node = nullptr;
  NeuronModel_addOperand(model, &int32Type);  // Operand 9: fuse
  fuse_node = graph->Add(x_name + "_pool_fuse", dims_int32);

  // Add output tensor type
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
    out_node = graph->Get(out_name);
  } else {
    NeuronModel_addOperand(model, &outType);
    out_node = graph->Add(out_name, dims_out);
  }
  VLOG(3) << "output_scale: " << out_scale
          << ", outType: " << outType.dimensions[0] << ":"
          << outType.dimensions[1] << ":" << outType.dimensions[2] << ":"
          << outType.dimensions[3];

  // Add padding value
  int32_t padding_val[1];
  padding_val[0] = paddings[2];
  NeuronModel_setOperandValue(
      model, paddingL_node->index(), padding_val, sizeof(int32_t) * 1);
  padding_val[0] = paddings[3];
  NeuronModel_setOperandValue(
      model, paddingR_node->index(), padding_val, sizeof(int32_t) * 1);
  padding_val[0] = paddings[0];
  NeuronModel_setOperandValue(
      model, paddingT_node->index(), padding_val, sizeof(int32_t) * 1);
  padding_val[0] = paddings[1];
  NeuronModel_setOperandValue(
      model, paddingB_node->index(), padding_val, sizeof(int32_t) * 1);

  // Add Stride
  int32_t stride_val[1];
  stride_val[0] = strides[1];  // Entry 1: width stride
  NeuronModel_setOperandValue(
      model, strideW_node->index(), stride_val, sizeof(int32_t) * 1);
  stride_val[0] = strides[0];  // Entry 0: height stride
  NeuronModel_setOperandValue(
      model, strideH_node->index(), stride_val, sizeof(int32_t) * 1);

  // Add filter
  int32_t filter_val[1];
  filter_val[0] =
      global_pooling ? x_dims[3] : ksize[1];  // Entry 1: filter width
  NeuronModel_setOperandValue(
      model, filterW_node->index(), filter_val, sizeof(int32_t) * 1);
  filter_val[0] =
      global_pooling ? x_dims[2] : ksize[0];  // Entry 0: filter height
  NeuronModel_setOperandValue(
      model, filterH_node->index(), filter_val, sizeof(int32_t) * 1);

  // Add fuse
  int32_t fuse_val[1] = {0};
  NeuronModel_setOperandValue(
      model, fuse_node->index(), fuse_val, sizeof(int32_t) * 1);

  std::vector<uint32_t> addInIndex = {x_node->index(),
                                      paddingL_node->index(),
                                      paddingR_node->index(),
                                      paddingT_node->index(),
                                      paddingB_node->index(),
                                      strideW_node->index(),
                                      strideH_node->index(),
                                      filterW_node->index(),
                                      filterH_node->index(),
                                      fuse_node->index()};
  std::vector<uint32_t> addOutIndex = {out_node->index()};

  int neuron_errCode;
  if (pooling_type == "max") {
    neuron_errCode = NeuronModel_addOperation(model,
                                              NEURON_MAX_POOL_2D,
                                              addInIndex.size(),
                                              &addInIndex[0],
                                              addOutIndex.size(),
                                              &addOutIndex[0]);
  } else {
    neuron_errCode = NeuronModel_addOperation(model,
                                              NEURON_AVERAGE_POOL_2D,
                                              addInIndex.size(),
                                              &addInIndex[0],
                                              addOutIndex.size(),
                                              &addOutIndex[0]);
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pool2d,
                         kAPU,
                         paddle::lite::subgraph::apu::PoolConverter);
