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

int ActConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite_metal::Tensor>();
  auto x_dims = x->dims();
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite_metal::Tensor>();
  auto output_dims = output->dims();
  const int64_t* x_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
  const int64_t* output_shape_data =
      const_cast<const int64_t*>(&output_dims.data()[0]);
  std::vector<int32_t> i_x_shape_data(x_dims.size());
  std::vector<int32_t> i_output_shape_data(output_dims.size());

  VLOG(3) << "[RKNPU] Converting " + op_type + "...";

  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));

  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));

  auto x_scale_name = "X0_scale";
  auto out_scale_name = "Out0_scale";
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  int bit_length = 8;
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

  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(x_shape_data[i]);
  }
  for (size_t i = 0; i < output_dims.size(); i++) {
    i_output_shape_data[i] = static_cast<int>(output_shape_data[i]);
  }
  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_var_name)) {
    x_node = graph->Get(x_var_name);
  } else {
    QuantizationInfo qnt;
    qnt.enable_int8 = enable_int8;
    if (enable_int8) {
      qnt.scale.push_back(input_scale);
      qnt.quant_bits = bit_length;
    }
    x_node = graph->Add(x_var_name, *x, precision, x_type->layout(), qnt);
  }

  std::shared_ptr<Node> output_node = nullptr;
  QuantizationInfo output_qnt;

  output_qnt.enable_int8 = enable_int8;

  if (enable_int8) {
    output_qnt.quant_bits = bit_length;
    output_qnt.scale.push_back(output_scale);
    output->mutable_data<int8_t>();
  }
  output_node = graph->Add(
      output_var_name, *output, precision, out_type->layout(), output_qnt);

  auto rGraph = graph->GetHandle();
  std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;
  std::vector<std::shared_ptr<rk::nn::Tensor>> outputs;

  inputs.push_back(x_node->data());
  outputs.push_back(output_node->data());
  if (op_type == "relu") {
    auto relu = rGraph->AddOperator(
        rk::nn::OperatorType::RELU, inputs, outputs, nullptr);
  } else if (op_type == "sigmoid") {
    auto sigmoid = rGraph->AddOperator(
        rk::nn::OperatorType::SIGMOID, inputs, outputs, nullptr);
  } else if (op_type == "relu6") {
    auto relu6 = rGraph->AddOperator(
        rk::nn::OperatorType::RELU6, inputs, outputs, nullptr);
  } else {
    LOG(WARNING) << "[RK-NPU] only support relu and sigmod, "
                    "but the activation type is "
                 << op_type;
    return FAILED;
  }
  return SUCCESS;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(relu,
                         kRKNPU,
                         paddle::lite_metal::subgraph::rknpu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(sigmoid,
                         kRKNPU,
                         paddle::lite_metal::subgraph::rknpu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(relu6,
                         kRKNPU,
                         paddle::lite_metal::subgraph::rknpu::ActConverter);
