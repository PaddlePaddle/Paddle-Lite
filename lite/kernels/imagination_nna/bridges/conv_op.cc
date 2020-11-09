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

#include "lite/operators/conv_op.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/imagination_nna/bridges/graph.h"
#include "lite/kernels/imagination_nna/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace imagination_nna {

int ConvConverter(void *ctx, OpLite *op, KernelBase *kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph *>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NNA] Converting " << op_type << "... ";

  CHECK(op_info->HasAttr("enable_int8") &&
        op_info->GetAttr<bool>("enable_int8"));

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();

  auto filter_name = op_info->Input("Filter").front();
  auto filter_scale_name = "Filter0_scale";
  auto filter = scope->FindMutableTensor(filter_name);
  auto filter_dims = filter->dims();

  auto output_name = op_info->Output("Output").front();
  auto output_scale_name = "Output0_scale";
  auto output = scope->FindMutableTensor(output_name);
  auto output_dims = output->dims();

  auto bs = input_dims[0];
  auto ic = input_dims[1];
  auto oc = filter_dims[0];
  CHECK_EQ(input_dims.size(), 4L);
  CHECK_EQ(output_dims.size(), 4L);
  CHECK_EQ(filter_dims.size(), 4L);
  CHECK_EQ(output_dims[0], bs);
  CHECK_EQ(output_dims[1], oc);
  std::vector<int> strides = op_info->GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  std::vector<int> dilations = op_info->GetAttr<std::vector<int>>("dilations");
  bool with_act =
      op_info->HasAttr("with_act") && op_info->GetAttr<bool>("with_act");
  std::string act_type =
      with_act ? op_info->GetAttr<std::string>("act_type") : "";
  float leaky_relu_alpha = act_type == "leaky_relu"
                               ? op_info->GetAttr<float>("leaky_relu_alpha")
                               : 0.f;

  CHECK_EQ(strides.size(), 2L);
  CHECK_EQ(dilations.size(), 2L);

  CHECK(op_info->HasInputScale(input_scale_name, true));
  float input_scale = op_info->GetInputScale(input_scale_name, true)[0];
  CHECK(op_info->HasInputScale(filter_scale_name, true));
  std::vector<float> weight_scale =
      op_info->GetInputScale(filter_scale_name, true);
  CHECK(op_info->HasOutputScale(output_scale_name, true));
  float output_scale = op_info->GetOutputScale(output_scale_name, true)[0];

  TensorInfo qnt;

  // Input node
  std::shared_ptr<Node> input_node = nullptr;
  imgdnn_tensor in_tensor;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
    in_tensor = input_node->data();
  } else {
    TensorInfoReset(&qnt);
    qnt.type = IMGDNN_TYPE_Q_U8;
    qnt.scales.push_back(input_scale);
    qnt.zero_points.push_back(128);
    input_node = graph->Add(input_name, *input, qnt, Node::Role::kInput);
    in_tensor = input_node->data();
  }

  if (paddings.size() == 2L) {
    for (size_t i = 0; i < strides.size(); ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "[NNA] Paddings size should be the same or twice as the input size.";

  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  operators::UpdatePaddingAndDilation(&paddings,
                                      &dilations,
                                      strides,
                                      padding_algorithm,
                                      input_dims,
                                      filter_dims);

  // Check depthwise mode, and decide whether use ConvolutionDepthwise Op
  bool is_depthwise_mode = (ic == groups && oc == groups && groups != 1);

  // Filter node
  bool per_channel = isScalesPerChannel(weight_scale);
  TensorInfoReset(&qnt);
  uint8_t *weights_u8 =
      graph->GetBuilder()->GetBufromPool(filter_dims.production());
  char *weight_src = static_cast<char *>(filter->raw_data());

  qnt.type = IMGDNN_TYPE_Q_U8;
  if (per_channel) {
    qnt.scales.assign(weight_scale.begin(), weight_scale.end());
    qnt.zero_points.assign(weight_scale.size(), 128);
    qnt.count = oc;
    qnt.axis = 1;
  } else {
    qnt.scales.push_back(weight_scale.at(0));
    qnt.zero_points.push_back(128);
  }
  for (int i = 0; i < filter_dims.production(); i++) {
    weights_u8[i] = static_cast<uint8_t>(weight_src[i] + 128);
  }

  std::shared_ptr<Node> filter_node = graph->Add(filter_name,
                                                 weights_u8,
                                                 filter_dims.Vectorize(),
                                                 qnt,
                                                 Node::Role::kConst);
  imgdnn_tensor filter_tensor = filter_node->data();

  // Add bias node if exists bias
  // Supports the bias nodes with the following dimensions
  // 0: {oc}
  // 1: {1, oc, oh, ow}
  // 2: {n, oc, oh, ow}
  std::shared_ptr<Node> bias_node;
  imgdnn_tensor bias_tensor = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias = scope->FindMutableTensor(bias_name);
      auto bias_dims = bias->dims();
      auto bias_data_size = bias_dims.production();
      auto output_data_size = output_dims.production();
      std::vector<int64_t> bias_shape;
      if (bias_data_size == oc) {
        // 0: {oc}
        bias_shape = {1, oc, 1, 1};
      } else if (bias_data_size == output_data_size / bs) {
        // 1: {1, oc, oh, ow}
        bias_shape = {1, output_dims[1], output_dims[2], output_dims[3]};
      } else if (bias_data_size == output_data_size) {
        // 2: {n, oc, oh, ow}
        bias_shape = output_dims.Vectorize();
      } else {
        LOG(WARNING)
            << "[NNA] Bias dimension " << bias_dims
            << " isn't supported in conv2d Op when output dimension is "
            << output_dims;
        return FAILED;
      }

      TensorInfoReset(&qnt);
      qnt.type = IMGDNN_TYPE_I32;
      if (per_channel) {
        qnt.scales.resize(bias_data_size);
        for (int i = 0; i < bias_data_size; i++)
          qnt.scales[i] = input_scale * weight_scale[i];
        qnt.zero_points.assign(bias_data_size, 0);
        qnt.count = 2;
        qnt.axis = 1;
      } else {
        qnt.scales.push_back(input_scale * weight_scale[0]);
        qnt.zero_points.push_back(0);
      }

      int quant_bits = 32;
      auto dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);
      auto dtype_min = static_cast<int>(0 - dtype_max);

      auto bias_data = bias->data<float, float>();
      int32_t *bias_qnt_data =
          reinterpret_cast<int32_t *>(graph->GetBuilder()->GetBufromPool(
              bias_dims.production() * sizeof(int32_t)));
      for (int i = 0; i < bias_data_size; i++) {
        float current_scale = per_channel ? qnt.scales[i] : qnt.scales[0];
        bias_qnt_data[i] = std::min(
            std::max(static_cast<int>(bias_data[i] / current_scale), dtype_min),
            dtype_max);
      }

      std::vector<int64_t> shapes{1, oc};
      bias_node =
          graph->Add(bias_name, bias_qnt_data, shapes, qnt, Node::Role::kConst);

      bias_tensor = bias_node->data();
    }
  }

  unsigned int img_stride[2] = {static_cast<unsigned int>(strides[0]),
                                static_cast<unsigned int>(strides[1])};
  // top,left
  unsigned int pad_to_begin[2] = {static_cast<unsigned int>(paddings[0]),
                                  static_cast<unsigned int>(paddings[2])};
  // bottom,right
  unsigned int pad_to_end[2] = {static_cast<unsigned int>(paddings[1]),
                                static_cast<unsigned int>(paddings[3])};
  unsigned int img_dilation[2] = {static_cast<unsigned int>(dilations[0]),
                                  static_cast<unsigned int>(dilations[1])};

  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_scale;
  output_quant_param.zero_point = 128;

  imgdnn_tensor conv_out =
      graph->GetBuilder()->CreateConvolutionLayer(in_tensor,
                                                  filter_tensor,
                                                  bias_tensor,
                                                  output_quant_param,
                                                  img_stride,
                                                  pad_to_begin,
                                                  pad_to_end,
                                                  img_dilation,
                                                  is_depthwise_mode);

  if (!act_type.empty()) {
    imgdnn_tensor act_out;
    if (act_type == "leaky_relu") {
      act_out = graph->GetBuilder()->CreateReLULayer(
          conv_out, false, 0.0, false, 0.0, leaky_relu_alpha);
    } else if (act_type == "relu6") {
      act_out = graph->GetBuilder()->CreateReLULayer(
          conv_out, true, 0.0, true, 6.0, false);
    } else if (act_type == "relu") {
      act_out = graph->GetBuilder()->CreateReLULayer(
          conv_out, true, 0.0, false, 0.0, false);
    } else {
      VLOG(3) << "act_type: " << act_type << " Not handled";
    }
    graph->Add(output_name, act_out, IMGDNN_TYPE_Q_U8);
  } else {
    graph->Add(output_name, conv_out, IMGDNN_TYPE_Q_U8);
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    conv2d,
    kImaginationNNA,
    paddle::lite::subgraph::imagination_nna::ConvConverter);

REGISTER_SUBGRAPH_BRIDGE(
    depthwise_conv2d,
    kImaginationNNA,
    paddle::lite::subgraph::imagination_nna::ConvConverter);
