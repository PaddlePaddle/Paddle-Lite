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

#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/rknpu/bridges/graph.h"
#include "lite/kernels/rknpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace rknpu {

int ConcatConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[RKNPU] Converting " << op_type << " ... ";

  // Get input and output vars and op attributes
  auto x_names = op_info->Input("X");
  auto out_name = op_info->Output("Out").front();
  auto output = scope->FindMutableTensor(out_name);

  auto axis = op_info->GetAttr<int>("axis");
  auto num = x_names.size();

  // for quantization
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  int bit_length = 8;
  DataLayoutType layout = DATALAYOUT(kNCHW);
  PrecisionType precision = PRECISION(kFloat);

  if (op_info->HasAttr("enable_int8")) {
    enable_int8 = op_info->GetAttr<bool>("enable_int8");
    bit_length = op_info->GetAttr<int>("bit_length");
    CHECK(op_info->HasOutputScale(out_name));
    output_scale = op_info->GetOutputScale(out_name)[0];

    if (enable_int8) {
      precision = PRECISION(kInt8);
    }
  }

  // Traverse all of input nodes which are added into the new created concat
  // node
  std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;
  std::vector<std::shared_ptr<rk::nn::Tensor>> outputs;

  int idx = 1;
  for (auto& x_name : x_names) {
    auto x = scope->FindMutableTensor(x_name);
    auto x_dims = x->dims();
    std::shared_ptr<Node> x_node = nullptr;
    if (graph->Has(x_name)) {
      x_node = graph->Get(x_name);
    } else {
      x_node = graph->Add(x_name, *x);
      QuantizationInfo qnt;
      qnt.enable_int8 = enable_int8;

      if (enable_int8) {
        CHECK(op_info->HasInputScale(x_name));
        input_scale = op_info->GetInputScale(x_name)[0];
        qnt.quant_bits = bit_length;
        qnt.scale.push_back(input_scale);
        x->mutable_data<int8_t>();
      }
      x_node = graph->Add(x_name, *x, precision, layout, qnt);
    }

    inputs.push_back(x_node->data());
    idx++;
  }

  std::shared_ptr<Node> output_node = nullptr;
  QuantizationInfo output_qnt;

  output_qnt.enable_int8 = enable_int8;

  if (enable_int8) {
    output_qnt.quant_bits = bit_length;
    output_qnt.scale.push_back(output_scale);
    output->mutable_data<int8_t>();
  }

  output_node = graph->Add(out_name, *output, precision, layout, output_qnt);
  outputs.push_back(output_node->data());

  rk::nn::ConcatAttr attrs;
  attrs.axis = axis;

  auto rGraph = graph->GetHandle();
  auto concat = rGraph->AddOperator(
      rk::nn::OperatorType::CONCAT, inputs, outputs, &attrs);

  return SUCCESS;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(concat,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::ConcatConverter);
