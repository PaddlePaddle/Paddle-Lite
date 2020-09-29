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

int FCConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[RKNPU] Converting " + op_type + "...";

  auto input_name = op_info->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  CHECK_GE(input_dims.size(), 2UL);
  auto w_name = op_info->Input("W").front();
  auto w_scale_name = "W0_scale";
  auto w = scope->FindMutableTensor(w_name);
  auto w_dims = w->dims();
  CHECK_EQ(w_dims.size(), 2UL);
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto output = scope->FindMutableTensor(out_name);
  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  int m = input_dims.Slice(0, in_num_col_dims).production();
  int k = input_dims.Slice(in_num_col_dims, input_dims.size()).production();
  int n = w_dims[1];
  CHECK_EQ(k * n, w_dims.production());
  VLOG(3) << "[RKNPU] input dims: " << input_dims << " w dims: " << w_dims
          << " m: " << m << " k: " << k << " n: " << n;

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

  // Create input node and reshape it to (m, k, 1, 1)
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }

  // Create w const node, set its shape to (n, k) and fill with
  // the transposed w tensor
  auto* transpose_w = scope->NewTensor(w_name + "/transpose");
  std::shared_ptr<Node> trans_w_node = nullptr;
  transpose_w->Resize({n, k});
  transpose_w->set_persistable(true);

  if (enable_int8) {
    QuantizationInfo filter_qnt;
    CHECK(op_info->HasInputScale(w_scale_name, true));
    auto weight_scale = op_info->GetInputScale(w_scale_name, true);
    filter_qnt.enable_int8 = enable_int8;
    filter_qnt.scale = weight_scale;
    filter_qnt.quant_bits = bit_length;

    auto transpose_w_data = transpose_w->mutable_data<int8_t>();
    auto w_data = w->mutable_data<int8_t>();

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        transpose_w_data[j * k + i] = w_data[i * n + j];
      }
    }
    trans_w_node =
        graph->Add(w_name, *transpose_w, precision, layout, filter_qnt);
  } else {
    auto transpose_w_data = transpose_w->mutable_data<float>();
    auto w_data = w->mutable_data<float>();

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        transpose_w_data[j * k + i] = w_data[i * n + j];
      }
    }
    trans_w_node = graph->Add(w_name, *transpose_w, precision, layout);
  }

  // Add bias node if bias tensor exists
  std::shared_ptr<Node> bias_node = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias = scope->FindMutableTensor(bias_name);
      auto bias_dims = bias->dims();
      auto bias_data_size = bias_dims.production();
      std::vector<int64_t> bias_shape = {n};

      VLOG(3) << "[RKNPU] bias precision: "
              << PrecisionToStr(bias->precision());
      // We need to quantize bias
      if (enable_int8) {
        auto bias_name_qnt = bias_name + "/qnt";
        auto* bias_qnt = scope->NewTensor(bias_name_qnt);
        CHECK(op_info->HasInputScale(w_scale_name, true));
        auto weight_scale = op_info->GetInputScale(w_scale_name, true);

        bias_qnt->Resize(bias_shape);
        bias_qnt->set_persistable(true);
        bias_qnt->set_precision(PrecisionType::kInt32);

        auto* bias_qnt_data = bias_qnt->mutable_data<int32_t>();
        auto* bias_data = bias->mutable_data<float>();

        QuantizationInfo qnt;
        qnt.enable_int8 = enable_int8;
        qnt.quant_bits = 32;
        qnt.scale.resize(weight_scale.size());

        for (int i = 0; i < weight_scale.size(); i++) {
          qnt.scale[i] = input_scale * weight_scale[i];
        }

        auto dtype_max = static_cast<int>((1 << (qnt.quant_bits - 1)) - 1);
        auto dtype_min = static_cast<int>(0 - dtype_max);

        for (int i = 0; i < n; i++) {
          bias_qnt_data[i] =
              std::min(std::max(static_cast<int>(bias_data[i] / qnt.scale[i]),
                                dtype_min),
                       dtype_max);
        }

        bias_node = graph->Add(
            bias_name, *bias_qnt, bias_qnt->precision(), layout, qnt);
      } else {
        bias_node = graph->Add(bias_name, *bias, bias_shape);
      }
    }
  } else {
    auto bias_name = w_name + "/bias/dummy";
    auto* bias = scope->NewTensor(bias_name);
    std::vector<int64_t> bias_shape = {n};

    bias->Resize(bias_shape);
    bias->set_persistable(true);

    if (enable_int8) {
      CHECK(op_info->HasInputScale(w_scale_name, true));
      auto weight_scale = op_info->GetInputScale(w_scale_name, true);
      bias->set_precision(PrecisionType::kInt32);
      auto* bias_data = bias->mutable_data<int32_t>();

      for (int i = 0; i < n; i++) {
        bias_data[i] = 0;
      }

      QuantizationInfo qnt;
      qnt.enable_int8 = enable_int8;
      qnt.quant_bits = 32;
      qnt.scale.resize(weight_scale.size());

      for (int i = 0; i < weight_scale.size(); i++) {
        qnt.scale[i] = input_scale * weight_scale[i];
      }

      bias_node = graph->Add(bias_name, *bias, bias->precision(), layout, qnt);
    } else {
      bias->set_precision(PrecisionType::kFloat);
      auto* bias_data = bias->mutable_data<float>();

      for (int i = 0; i < n; i++) {
        bias_data[i] = 0.0;
      }
      bias_node = graph->Add(bias_name, *bias, bias_shape);
    }
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

  inputs.push_back(input_node->data());
  inputs.push_back(trans_w_node->data());
  inputs.push_back(bias_node->data());
  outputs.push_back(output_node->data());

  rk::nn::FCAttr attrs;
  attrs.weights = n;
  attrs.has_relu = false;

  auto rGraph = graph->GetHandle();
  auto fc = rGraph->AddOperator(
      rk::nn::OperatorType::FULLCONNECT, inputs, outputs, &attrs);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fc,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::FCConverter);
