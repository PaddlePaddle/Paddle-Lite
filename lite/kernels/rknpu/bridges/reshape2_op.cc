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
#include "lite/operators/reshape_op.h"

namespace paddle {
namespace lite_metal {
namespace subgraph {
namespace rknpu {

int Reshape2Converter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  // Read shape from "ShapeTensor"(input), or "Shape"(input), or "shape"(attr)
  // NOW ONLY SUPPORT "shape"(attr)
  rk::nn::ReshapeAttr attrs;  // NCHW is fit well with RKchip
  auto shape = op_info->GetAttr<std::vector<int>>("shape");
  auto origin_out_shape = lite_metal::operators::ValidateShape(shape, x_dims);
  std::vector<int64_t> out_shape;

  for (int i = 0; i < origin_out_shape.size(); i++) {
    out_shape.push_back(origin_out_shape[i]);
  }
  if (out_shape.size() > 4) {
    LOG(WARNING) << "[RK-NPU] only supports less than 4 dimensions, "
                    "but shape has "
                 << out_shape.size();
    return FAILED;
  }
  for (int i = 0; i < origin_out_shape.size(); i++) {
    attrs.shapes.push_back(out_shape[i]);
  }

  // All done (inputs, outputs, attrs)
  auto rGraph = graph->GetHandle();
  auto reshape2 = rGraph->AddOperator(
      rk::nn::OperatorType::RESHAPE, inputs, outputs, &attrs);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(reshape2,
                         kRKNPU,
                         paddle::lite_metal::subgraph::rknpu::Reshape2Converter);
