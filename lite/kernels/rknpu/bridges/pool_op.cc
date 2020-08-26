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
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/rknpu/bridges/graph.h"
#include "lite/kernels/rknpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace rknpu {

int PoolConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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

  auto out_name = op_info->Output("Out").front();
  auto output = scope->FindMutableTensor(out_name);

  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");

  // for quantization
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  int bit_length = 8;
  DataLayoutType layout = DATALAYOUT(kNCHW);
  PrecisionType precision = PRECISION(kFloat);

  if (x->precision() == PRECISION(kInt8)) {
    // enable_int8 = op_info->GetAttr<bool>("enable_int8");
    enable_int8 = true;
    CHECK(op_info->HasInputScale(x_name));
    input_scale = op_info->GetInputScale(x_name)[0];
    bit_length = op_info->GetAttr<int>("bit_length");
    CHECK(op_info->HasOutputScale(out_name));
    output_scale = op_info->GetOutputScale(out_name)[0];

    if (enable_int8) {
      precision = PRECISION(kInt8);
      LOG(WARNING) << "[RKNPU] Pooling int8";
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
      qnt.scale.push_back(input_scale);
      qnt.quant_bits = bit_length;
    }
    x_node = graph->Add(x_name, *x, x->precision(), layout, qnt);
  }

  // pool mode
  rk::nn::PoolType mode = rk::nn::PoolType::POOLING_UNKNOWN;
  if (pooling_type == "max") {
    mode = rk::nn::PoolType::POOLING_MAX;
  } else if (pooling_type == "avg") {
    mode = rk::nn::PoolType::POOLING_AVG;
  } else {
    LOG(WARNING) << "[RKNPU] Unsupported pooling type: " << pooling_type;
    return FAILED;
  }

  // pad mode
  rk::nn::PadType pad_mode = rk::nn::PadType::AUTO;
  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  if (padding_algorithm == "SAME") {
    pad_mode = rk::nn::PadType::SAME;
  } else if (padding_algorithm == "VALID") {
    pad_mode = rk::nn::PadType::VALID;
  }

  // paddings and strides
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < 2L; ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "[NPU] Paddings size should be the same or twice as the inputs size.";

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

  // ceil mode
  int ceil_mode = 0;
  if (op_info->HasAttr("ceil_mode")) {
    ceil_mode = op_info->GetAttr<bool>("ceil_mode") ? 1 : 0;
  }

  QuantizationInfo output_qnt;
  output_qnt.enable_int8 = enable_int8;
  if (enable_int8) {
    output_qnt.quant_bits = bit_length;
    output_qnt.scale.push_back(output_scale);
    output->mutable_data<int8_t>();
  }

  auto output_node =
      graph->Add(out_name, *output, precision, layout, output_qnt);

  std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;
  std::vector<std::shared_ptr<rk::nn::Tensor>> outputs;

  inputs.push_back(x_node->data());
  outputs.push_back(output_node->data());

  rk::nn::PoolAttr attrs;
  attrs.ksize[0] = ksize[0];
  attrs.ksize[1] = ksize[1];
  attrs.stride[0] = strides[0];
  attrs.stride[1] = strides[1];
  attrs.pad[0] = paddings[0];
  attrs.pad[1] = paddings[1];
  attrs.pad[2] = paddings[2];
  attrs.pad[3] = paddings[3];
  attrs.pad_type = pad_mode;
  attrs.pool_type = mode;
  attrs.global_pooling = global_pooling;

  if (ceil_mode) {
    attrs.round_type = rk::nn::RoundType::ROUND_CEIL;
  } else {
    attrs.round_type = rk::nn::RoundType::ROUND_FLOOR;
  }

  auto rGraph = graph->GetHandle();
  auto pool =
      rGraph->AddOperator(rk::nn::OperatorType::POOL, inputs, outputs, &attrs);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pool2d,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::PoolConverter);
