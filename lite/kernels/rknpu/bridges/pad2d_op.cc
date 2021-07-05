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
namespace lite_metal {
namespace subgraph {
namespace rknpu {

int Pad2dConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  auto x_rank = x_dims.size();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto output = scope->FindMutableTensor(out_name);

  // For quantization
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  int bit_length = 8;
  DataLayoutType layout = DATALAYOUT(kNCHW);
  PrecisionType precision = PRECISION(kFloat);

  if (op_info->HasInputScale(x_scale_name, true) &&
      op_info->HasOutputScale(out_scale_name, true)) {
    enable_int8 = true;
    input_scale = op_info->GetInputScale(x_scale_name, true)[0];
    bit_length = op_info->GetAttr<int>("bit_length");
    output_scale = op_info->GetOutputScale(out_scale_name, true)[0];
    precision = PRECISION(kInt8);
  } else {
    enable_int8 = false;
    LOG(WARNING) << "[RK-NPU] the op is float-type " << op_type;
    precision = PRECISION(kFloat);
  }
  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    QuantizationInfo qnt;
    qnt.enable_int8 = enable_int8;
    if (enable_int8) {
      qnt.scale.push_back(input_scale);
      qnt.quant_bits = bit_length;
    }
    x_node = graph->Add(x_name, *x, precision, layout, qnt);
  }

  // Output Node
  std::shared_ptr<Node> output_node = nullptr;
  QuantizationInfo output_qnt;
  output_qnt.enable_int8 = enable_int8;
  if (enable_int8) {
    output_qnt.quant_bits = bit_length;
    output_qnt.scale.push_back(output_scale);
    output->mutable_data<int8_t>();
  }
  output_node = graph->Add(out_name, *output, precision, layout, output_qnt);

  // fill inputs&outputs with node
  std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;
  std::vector<std::shared_ptr<rk::nn::Tensor>> outputs;
  inputs.push_back(x_node->data());
  outputs.push_back(output_node->data());

  // OP option
  rk::nn::PadAttr attr;
  auto mode = op_info->GetAttr<std::string>("mode");
  if (mode == "constant") {
    attr.mode = rk::nn::PadMode::PAD_CONSTANT;
  } else if (mode == "reflect") {
    attr.mode = rk::nn::PadMode::PAD_REFLECT;
  } else if (mode == "edge") {
    attr.mode = rk::nn::PadMode::PAD_REPLICATE;
  } else {
    LOG(WARNING) << "[RKNPU] pad mode " << mode << " isn't supported in RK-NPU";
    return FAILED;
  }

  auto padding = op_info->GetAttr<std::vector<int>>("paddings");
  int pads_num = padding.size() / 2;
  for (int i = 0; i < x_rank; i++) {
    if (i < pads_num) {
      attr.begin.push_back(padding[2 * i]);
      attr.end.push_back(padding[2 * i + 1]);
    } else {
      attr.begin.push_back(0);
      attr.end.push_back(0);
    }
  }
  std::reverse(attr.begin.begin(), attr.begin.end());
  std::reverse(attr.end.begin(), attr.end.end());
  auto pad_value = op_info->GetAttr<float>("pad_value");
  attr.const_val = pad_value;

  // All done (inputs, outputs, attr)
  auto rGraph = graph->GetHandle();
  auto pad2d =
      rGraph->AddOperator(rk::nn::OperatorType::PAD, inputs, outputs, &attr);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pad2d,
                         kRKNPU,
                         paddle::lite_metal::subgraph::rknpu::Pad2dConverter);
