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
#include "lite/kernels/rknpu/bridges/graph.h"
#include "lite/kernels/rknpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace rknpu {

std::vector<int64_t> CvtYShape(const DDim& x_dims,
                               const DDim& y_dims,
                               int axis) {
  CHECK_EQ(x_dims.size(), 4UL) << "[RKNPU] Only support 4-dimension x";
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

int ElementwiseConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[RKNPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto y_name = op_info->Input("Y").front();
  auto y_scale_name = "Y0_scale";
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto out_type = kernel->GetOutputDeclType("Out");
  auto output = scope->FindMutableTensor(out_name);
  auto axis = op_info->GetAttr<int>("axis");

  // for quantization
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  int bit_length = 8;
  DataLayoutType layout = DATALAYOUT(kNCHW);
  PrecisionType precision = PRECISION(kFloat);

  if (op_info->HasAttr("enable_int8")) {
    enable_int8 = op_info->GetAttr<bool>("enable_int8");
    CHECK(op_info->HasInputScale(x_scale_name, true));
    input_scale = op_info->GetInputScale(x_scale_name, true)[0];
    bit_length = op_info->GetAttr<int>("bit_length");
    CHECK(op_info->HasOutputScale(out_scale_name, true));
    output_scale = op_info->GetOutputScale(out_scale_name, true)[0];

    if (enable_int8) {
      precision = PRECISION(kInt8);
    }
  }

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    QuantizationInfo qnt;
    qnt.enable_int8 = enable_int8;

    if (enable_int8) {
      qnt.scale.clear();
      qnt.scale.push_back(input_scale);
      qnt.quant_bits = op_info->GetAttr<int>("bit_length");
    }
    x_node = graph->Add(x_name, *x, precision, layout, qnt);
  }

  // Y node
  std::shared_ptr<Node> y_node = nullptr;
  if (graph->Has(y_name)) {
    y_node = graph->Get(y_name);
  } else {
    // auto y_new_shape = CvtYShape(x_dims, y_dims, axis);
    // y_node = graph->Add(y_name, *y, y_new_shape);
    QuantizationInfo qnt;
    qnt.enable_int8 = enable_int8;

    if (enable_int8) {
      qnt.quant_bits = bit_length;
      qnt.scale.clear();
      qnt.scale.push_back(input_scale);
    }
    y_node = graph->Add(y_name, *y, precision, layout, qnt);
  }

  std::shared_ptr<Node> output_node = nullptr;
  QuantizationInfo output_qnt;

  output_qnt.enable_int8 = enable_int8;

  if (enable_int8) {
    output_qnt.quant_bits = bit_length;
    output_qnt.scale.clear();
    output_qnt.scale.push_back(output_scale);
    output->mutable_data<int8_t>();
  }

  output_node = graph->Add(out_name, *output, precision, layout, output_qnt);

  std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;
  std::vector<std::shared_ptr<rk::nn::Tensor>> outputs;

  inputs.push_back(x_node->data());
  inputs.push_back(y_node->data());
  outputs.push_back(output_node->data());

  auto rGraph = graph->GetHandle();

  // Elementwise node
  if (op_type == "elementwise_add") {
    auto elt_node = rGraph->AddOperator(
        rk::nn::OperatorType::ADD, inputs, outputs, nullptr);
  } else if (op_type == "elementwise_sub") {
    auto elt_node = rGraph->AddOperator(
        rk::nn::OperatorType::SUBTRACT, inputs, outputs, nullptr);
  } else if (op_type == "elementwise_mul") {
    auto elt_node = rGraph->AddOperator(
        rk::nn::OperatorType::MULTIPLY, inputs, outputs, nullptr);
  } else if (op_type == "elementwise_div") {
    auto elt_node = rGraph->AddOperator(
        rk::nn::OperatorType::DIVIDE, inputs, outputs, nullptr);
  } else {
    LOG(WARNING) << "[RKNPU] Unsupported op type: " << op_type;
    return FAILED;
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(elementwise_add,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_sub,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_mul,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_div,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::ElementwiseConverter);
