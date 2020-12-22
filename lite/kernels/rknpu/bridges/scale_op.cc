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

int ScaleConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[RKNPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto x_rank = x_dims.size();
  auto input_scale_name = "X0_scale";
  auto out_name = op_info->Output("Out").front();
  auto output = scope->FindMutableTensor(out_name);
  auto out_type = kernel->GetOutputDeclType("Out");
  auto out_scale_name = "Out0_scale";

  float scale = op_info->GetAttr<float>("scale");
  float bias = op_info->GetAttr<float>("bias");
  bool bias_after_scale = op_info->GetAttr<bool>("bias_after_scale");
  if (!bias_after_scale) {
    bias *= scale;
  }

  // for quantization
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  int bit_length = 8;
  DataLayoutType layout = DATALAYOUT(kNCHW);
  PrecisionType precision = PRECISION(kFloat);

  if (op_info->HasAttr("enable_int8")) {
    enable_int8 = op_info->GetAttr<bool>("enable_int8");
    CHECK(op_info->HasInputScale(input_scale_name, true));
    input_scale = op_info->GetInputScale(input_scale_name, true)[0];
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
      qnt.quant_bits = bit_length;
      qnt.scale.push_back(input_scale);
      x->mutable_data<int8_t>();
    }
    x_node = graph->Add(x_name, *x, precision, x_type->layout(), qnt);
  }

  // Scale node
  QuantizationInfo output_qnt;
  output_qnt.enable_int8 = enable_int8;

  if (enable_int8) {
    output_qnt.quant_bits = bit_length;
    output_qnt.scale.push_back(output_scale);
    output->mutable_data<int8_t>();
  }
  auto output_node =
      graph->Add(out_name, *output, precision, out_type->layout(), output_qnt);

  std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;
  std::vector<std::shared_ptr<rk::nn::Tensor>> outputs;

  inputs.push_back(x_node->data());
  outputs.push_back(output_node->data());

  rk::nn::ScaleAttr attrs;
  attrs.scale = scale;
  attrs.bias = bias;

  auto rGraph = graph->GetHandle();
  rGraph->AddOperator(rk::nn::OperatorType::SCALE, inputs, outputs, &attrs);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(scale,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::ScaleConverter);
