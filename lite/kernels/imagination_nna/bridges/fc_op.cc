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
#include "lite/kernels/imagination_nna/bridges/graph.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace imagination_nna {

int FCConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NNA] Converting " + op_type + "...";

  CHECK(op_info->HasAttr("enable_int8"));
  CHECK(op_info->GetAttr<bool>("enable_int8"));

  auto input_name = op_info->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  auto input = scope->FindTensor(input_name);
  auto input_dims = input->dims();

  auto weight_name = op_info->Input("W").front();
  auto weight_scale_name = "W0_scale";
  auto weights = scope->FindTensor(weight_name);
  auto w_dims = weights->dims();
  CHECK_EQ(w_dims.size(), 2UL);

  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto out = scope->FindTensor(out_name);
  auto out_dims = out->dims();

  // notes : m, input row
  //         k, input col
  //         n, weight col
  // input_dims : {1,1024,1,1}
  // in_num_col_dims : 1
  // m =1, k=1024,n=1000
  // w_dims : {1024,1000}
  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  int m = input_dims.Slice(0, in_num_col_dims).production();
  int k = input_dims.Slice(in_num_col_dims, input_dims.size()).production();
  int n = w_dims[1];
  CHECK_EQ(k * n, w_dims.production());
  VLOG(3) << "[NNA] input dims: " << input_dims << " w dims: " << w_dims
          << " m: " << m << " k: " << k << " n: " << n;

  CHECK(op_info->HasInputScale(input_scale_name, true));
  float input_scale = op_info->GetInputScale(input_scale_name, true)[0];
  CHECK(op_info->HasInputScale(weight_scale_name, true));
  std::vector<float> weight_scale =
      op_info->GetInputScale(weight_scale_name, true);
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  float output_scale = op_info->GetOutputScale(out_scale_name, true)[0];

  // Create input node and reshape it to (m, k, 1, 1)
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    LOG(FATAL) << "[NNA] input node: " << input_name << ", could not be found";
  }

  // weight tensor
  std::shared_ptr<Node> weight_node = nullptr;
  bool per_channel = isScalesPerChannel(weight_scale);
  uint8_t* weights_u8 = graph->GetBuilder()->GetBufromPool(w_dims.production());

  TensorInfo qnt;
  qnt.type = IMGDNN_TYPE_Q_U8;
  if (per_channel) {
    LOG(FATAL)
        << "[NNA] FC per-channel quantization is not supported for Mirage";
  } else {
    qnt.scales.push_back(weight_scale.at(0));
    qnt.zero_points.push_back(128);
  }
  const char* weight_src = static_cast<const char*>(weights->raw_data());
  for (int i = 0; i < w_dims.production(); i++)
    weights_u8[i] = static_cast<uint8_t>(weight_src[i] + 128);
  weight_node = graph->Add(
      weight_name, weights_u8, w_dims.Vectorize(), qnt, Node::Role::kConst);

  // Add bias node if bias tensor exists
  imgdnn_tensor bias_tensor = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    std::shared_ptr<Node> bias_node = nullptr;
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias = scope->FindTensor(bias_name);
      auto bias_dims = bias->dims();
      CHECK_EQ(bias_dims.production(), n);

      TensorInfoReset(&qnt);
      qnt.type = IMGDNN_TYPE_I32;
      if (per_channel) {
        qnt.scales.resize(weight_scale.size());
        qnt.count = bias_dims.size();
        qnt.axis = 0;
        for (int i = 0; i < weight_scale.size(); i++) {
          qnt.scales[i] = input_scale * weight_scale[i];
        }
        LOG(FATAL) << "[NNA] per-channel quantization is not supported for FC";
      } else {
        qnt.scales.push_back(weight_scale.at(0) * input_scale);
        qnt.zero_points.push_back(0);
      }

      int quant_bits = 32;
      auto dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);
      auto dtype_min = static_cast<int>(0 - dtype_max);

      auto* bias_data = bias->data<float, float>();
      int32_t* bias_qnt_data =
          reinterpret_cast<int32_t*>(graph->GetBuilder()->GetBufromPool(
              bias_dims.production() * sizeof(int32_t)));
      for (int i = 0; i < n; i++) {
        float current_scale = per_channel ? qnt.scales[i] : qnt.scales[0];
        bias_qnt_data[i] = std::min(
            std::max(static_cast<int>(bias_data[i] / current_scale), dtype_min),
            dtype_max);
      }

      std::vector<int64_t> shapes{1};
      bias_node =
          graph->Add(bias_name, bias_qnt_data, shapes, qnt, Node::Role::kConst);
    }
    bias_tensor = bias_node->data();
  }

  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_scale;
  output_quant_param.zero_point = 128;
  imgdnn_tensor fc_out_tensor = graph->GetBuilder()->CreateFullyConnectedLayer(
      input_node->data(), weight_node->data(), bias_tensor, output_quant_param);

  imgdnn_tensor_descriptor desc =
      graph->GetBuilder()->GetTensorDescriptor(fc_out_tensor);
  graph->Add(out_name, fc_out_tensor, desc.type);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fc,
                         kImaginationNNA,
                         paddle::lite::subgraph::imagination_nna::FCConverter);
