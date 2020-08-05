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

int BatchNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  auto scale_name = op_info->Input("Scale").front();
  auto scale = scope->FindMutableTensor(scale_name);
  auto bias_name = op_info->Input("Bias").front();
  auto bias = scope->FindMutableTensor(bias_name);
  auto mean_name = op_info->Input("Mean").front();
  auto mean = scope->FindMutableTensor(mean_name);
  auto variance_name = op_info->Input("Variance").front();
  auto variance = scope->FindMutableTensor(variance_name);
  auto y_name = op_info->Output("Y").front();
  auto y = scope->FindMutableTensor(y_name);
  float momentum = op_info->GetAttr<float>("momentum");
  float epsilon = op_info->GetAttr<float>("epsilon");
  int mode = 1;  // bnScale, bnBias tensor dims are 1xCx1x1
  bool use_global_stats = op_info->GetAttr<bool>("use_global_stats");

  // for quantization
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  int bit_length = 8;
  DataLayoutType layout = DATALAYOUT(kNCHW);
  PrecisionType precision = PRECISION(kFloat);

  if (op_info->HasAttr("enable_int8")) {
    enable_int8 = op_info->GetAttr<bool>("enable_int8");
    CHECK(op_info->HasInputScale(x_name));
    input_scale = op_info->GetInputScale(x_name)[0];
    bit_length = op_info->GetAttr<int>("bit_length");
    CHECK(op_info->HasOutputScale(y_name));
    output_scale = op_info->GetOutputScale(y_name)[0];

    if (enable_int8) {
      precision = PRECISION(kInt8);
    }
  }

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Scale, Bias, Mean, Variance node
  auto scale_node = graph->Add(scale_name, *scale);
  auto bias_node = graph->Add(bias_name, *bias);
  auto mean_node = graph->Add(mean_name, *mean);
  auto variance_node = graph->Add(variance_name, *variance);

  std::shared_ptr<Node> output_node = nullptr;
  QuantizationInfo output_qnt;

  output_qnt.enable_int8 = enable_int8;

  if (enable_int8) {
    output_qnt.quant_bits = bit_length;
    output_qnt.scale.push_back(output_scale);
    y->mutable_data<int8_t>();
  }

  output_node = graph->Add(y_name, *y, precision, layout, output_qnt);

  std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;
  std::vector<std::shared_ptr<rk::nn::Tensor>> outputs;

  inputs.push_back(x_node->data());
  inputs.push_back(mean_node->data());
  inputs.push_back(variance_node->data());
  inputs.push_back(scale_node->data());
  inputs.push_back(bias_node->data());
  outputs.push_back(output_node->data());

  rk::nn::BatchNormAttr attrs;
  attrs.eps = epsilon;

  auto rGraph = graph->GetHandle();
  auto bn = rGraph->AddOperator(
      rk::nn::OperatorType::BATCH_NORM, inputs, outputs, &attrs);

  return SUCCESS;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(batch_norm,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::BatchNormConverter);
