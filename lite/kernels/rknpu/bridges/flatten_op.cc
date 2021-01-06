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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/rknpu/bridges/graph.h"
#include "lite/kernels/rknpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace rknpu {

int FlattenConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[RKNPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto x_type = kernel->GetInputDeclType("X");
  auto x_scale_name = "X0_scale";

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto out_type = kernel->GetOutputDeclType("Out");
  auto out_scale_name = "Out0_scale";

  // for quantization
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  int bit_length = 8;
  DataLayoutType layout = DATALAYOUT(kNCHW);
  PrecisionType precision = x->precision();

  // x->set_precision(PRECISION(kInt8));

  VLOG(3) << "input shape is: " << x_dims.repr();
  VLOG(3) << "output shape is: " << out_dims.repr();
  VLOG(3) << "input precision is: " << PrecisionToStr(x_type->precision());
  VLOG(3) << "output precision is: " << PrecisionToStr(out_type->precision());
  VLOG(3) << "input tensor is: " << PrecisionToStr(x->precision());

  if (op_info->HasAttr("enable_int8")) {
    enable_int8 = op_info->GetAttr<bool>("enable_int8");
    bit_length = op_info->GetAttr<int>("bit_length");

    if (enable_int8) {
      CHECK(op_info->HasInputScale(x_scale_name, true));
      input_scale = op_info->GetInputScale(x_scale_name, true)[0];
      CHECK(op_info->HasOutputScale(out_scale_name, true));
      output_scale = op_info->GetOutputScale(out_scale_name, true)[0];
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
    out->mutable_data<int8_t>();
  }
  auto output_node =
      graph->Add(out_name, *out, precision, x_type->layout(), output_qnt);

  std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;
  std::vector<std::shared_ptr<rk::nn::Tensor>> outputs;

  inputs.push_back(x_node->data());
  outputs.push_back(output_node->data());

  rk::nn::ReshapeAttr attr;
  for (size_t i = 0; i < out_dims.size(); i++) {
    attr.shapes.push_back(static_cast<uint32_t>(out_dims[i]));
  }

  auto rGraph = graph->GetHandle();
  auto flatten = rGraph->AddOperator(
      rk::nn::OperatorType::RESHAPE, inputs, outputs, &attr);

  return SUCCESS;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(flatten,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::FlattenConverter);
REGISTER_SUBGRAPH_BRIDGE(flatten2,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::FlattenConverter);
