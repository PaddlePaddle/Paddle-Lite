// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int DeformableConvConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " << op_type << "... ";

  // Get input
  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();

  // Get offset
  auto offset_name = op_info->Input("Offset").front();
  auto offset = scope->FindMutableTensor(offset_name);
  auto offset_dims = offset->dims();

  // Get mask
  auto mask_name = op_info->Input("Mask").front();
  auto mask = scope->FindMutableTensor(mask_name);
  auto mask_dims = mask->dims();

  // Get output
  auto output_name = op_info->Output("Output").front();
  auto output = scope->FindMutableTensor(output_name);
  auto output_dims = output->dims();

  // get op attr
  auto filter_name = op_info->Input("Filter").front();
  auto filter = scope->FindMutableTensor(filter_name);
  auto filter_dims = filter->dims();

  auto bs = input_dims[0];
  auto oc = filter_dims[0];

  CHECK_EQ(input_dims.size(), 4L);
  CHECK_EQ(output_dims.size(), 4L);
  CHECK_EQ(filter_dims.size(), 4L);
  CHECK_EQ(output_dims[0], bs);
  CHECK_EQ(output_dims[1], oc);

  auto deformable_groups = op_info->GetAttr<int>("deformable_groups");
  std::vector<int> strides = op_info->GetAttr<std::vector<int>>("strides");
  std::vector<int> dilations = op_info->GetAttr<std::vector<int>>("dilations");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");

  CHECK_EQ(strides.size(), 2L);
  CHECK_EQ(dilations.size(), 2L);

  if (paddings.size() == 2L) {
    for (size_t i = 0; i < strides.size(); ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "[HUAWEI_ASCEND_NPU] Paddings size should be "
         "the same or twice as the input size.";

  // act
  bool with_act =
      op_info->HasAttr("with_act") && op_info->GetAttr<bool>("with_act");
  std::string act_type =
      with_act ? op_info->GetAttr<std::string>("act_type") : "";
  float leaky_relu_alpha = act_type == "leaky_relu"
                               ? op_info->GetAttr<float>("leaky_relu_alpha")
                               : 0.f;

  // Input node
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }

  // Deformable_offset mask node
  std::shared_ptr<Node> mask_node = nullptr;
  if (graph->Has(mask_name)) {
    mask_node = graph->Get(mask_name);
  } else {
    mask_node = graph->Add(mask_name, *mask);
  }

  // Deformable_offset offsets node
  std::shared_ptr<Node> input_offsets_node = nullptr;
  if (graph->Has(offset_name)) {
    input_offsets_node = graph->Get(offset_name);
  } else {
    input_offsets_node = graph->Add(offset_name, *offset);
  }

  // concat offsets and mask
  int num = 2;
  int axis = 1;
  auto concat_node = graph->Add<ge::op::ConcatD>(output_name + "/concat");
  auto concat_op = concat_node->data<ge::op::ConcatD>();
  concat_op->set_attr_concat_dim(axis);
  concat_op->set_attr_N(num);
  concat_op->create_dynamic_input_x(num);
  concat_op->set_dynamic_input_x(0, *mask_node->data());
  concat_op->set_dynamic_input_x(1, *input_offsets_node->data());
  DYNAMIC_INPUT_UPDATE(concat_op, x, 0, mask_node);
  DYNAMIC_INPUT_UPDATE(concat_op, x, 1, input_offsets_node);
  OUTPUT_UPDATE(concat_op, y, concat_node);

  // create deformable_offsets node and op
  auto deformable_offsets_node =
      graph->Add<ge::op::DeformableOffsets>(output_name);
  auto deformable_offsets_op =
      deformable_offsets_node->data<ge::op::DeformableOffsets>();
  deformable_offsets_op->set_input_x(*input_node->data());
  deformable_offsets_op->set_input_offsets(*concat_node->data());
  deformable_offsets_op->set_attr_strides(
      ge::Operator::OpListInt({1, 1, strides[0], strides[1]}));
  deformable_offsets_op->set_attr_pads(ge::Operator::OpListInt(
      {paddings[0], paddings[1], paddings[2], paddings[3]}));
  deformable_offsets_op->set_attr_ksize(
      ge::Operator::OpListInt({filter_dims[2], filter_dims[3]}));
  deformable_offsets_op->set_attr_dilations(
      ge::Operator::OpListInt({1, 1, dilations[0], dilations[1]}));
  deformable_offsets_op->set_attr_deformable_groups(deformable_groups);
  deformable_offsets_op->set_attr_data_format("NCHW");
  deformable_offsets_op->set_attr_modulated(true);

  INPUT_UPDATE(deformable_offsets_op, x, input_node);
  INPUT_UPDATE(deformable_offsets_op, offsets, concat_node);
  OUTPUT_UPDATE(deformable_offsets_op, y, deformable_offsets_node);

  // add reshape node to reshape output
  std::shared_ptr<Node> actual_shape_node = nullptr;
  actual_shape_node = graph->Add<int64_t>(output_name + "/shape",
                                          {output_dims[0],
                                           output_dims[1],
                                           output_dims[2] * filter_dims[2],
                                           output_dims[3] * filter_dims[3]});

  auto reshaped_offsets_output_node = graph->Add<ge::op::Reshape>(output_name);
  auto reshaped_offsets_output_op =
      reshaped_offsets_output_node->data<ge::op::Reshape>();
  reshaped_offsets_output_op->set_input_x(*deformable_offsets_node->data());
  reshaped_offsets_output_op->set_input_shape(*actual_shape_node->data());
  INPUT_UPDATE(reshaped_offsets_output_op, x, deformable_offsets_node);
  INPUT_UPDATE(reshaped_offsets_output_op, shape, actual_shape_node);
  OUTPUT_UPDATE(reshaped_offsets_output_op, y, reshaped_offsets_output_node);

  // conv2d filter node
  std::shared_ptr<Node> filter_node = nullptr;
  filter_node = graph->Add(filter_name, *filter);

  // Add conv2d bias node if exists bias
  // Supports the bias nodes with the following dimensions
  // 0: {oc} => 1D tensor of foramt ND
  // 1: {1, oc, oh, ow}
  // 2: {n, oc, oh, ow}
  std::vector<int64_t> bias_shape;
  std::shared_ptr<Node> bias_node = nullptr;
  bool is_channel_bias = false;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias = scope->FindMutableTensor(bias_name);
      auto bias_dims = bias->dims();
      auto bias_data_size = bias_dims.production();
      auto output_data_size = output_dims.production();
      if (bias_data_size == oc) {
        // 0: {oc}
        bias_shape = {oc};
        is_channel_bias = true;
      } else if (bias_data_size == output_data_size / bs) {
        // 1: {1, oc, oh, ow}
        bias_shape = {1, output_dims[1], output_dims[2], output_dims[3]};
      } else if (bias_data_size == output_data_size) {
        // 2: {n, oc, oh, ow}
        bias_shape = output_dims.Vectorize();
      } else {
        LOG(WARNING)
            << "[HUAWEI_ASCEND_NPU] Bias dimension " << bias_dims
            << " isn't supported in conv2d Op when output dimension is "
            << output_dims;
        return FAILED;
      }
      bias_node = graph->Add(bias_name, *bias, bias_shape);
    }
  }

  // Conv2d node
  auto conv_node = graph->Add<ge::op::Conv2D>(output_name);
  auto conv_op = conv_node->data<ge::op::Conv2D>();
  conv_op->set_input_x(*reshaped_offsets_output_node->data());
  conv_op->set_input_filter(*filter_node->data());
  conv_op->set_attr_strides(
      ge::Operator::OpListInt({1, 1, filter_dims[2], filter_dims[3]}));
  conv_op->set_attr_pads(ge::Operator::OpListInt({0, 0, 0, 0}));
  conv_op->set_attr_dilations(ge::Operator::OpListInt({1, 1, 1, 1}));
  conv_op->set_attr_data_format("NCHW");

  if (bias_node != nullptr && is_channel_bias) {
    conv_op->set_input_bias(*bias_node->data());
    INPUT_UPDATE(conv_op, bias, bias_node);
  }

  INPUT_UPDATE(conv_op, x, reshaped_offsets_output_node);
  INPUT_UPDATE(conv_op, filter, filter_node);
  OUTPUT_UPDATE(conv_op, y, conv_node);

  // append Add node to support bias
  if (bias_node != nullptr && !is_channel_bias) {
    auto add_node = graph->Add<ge::op::Add>(output_name);
    auto add_op = add_node->data<ge::op::Add>();
    add_op->set_input_x1(*conv_node->data());
    add_op->set_input_x2(*bias_node->data());
    INPUT_UPDATE(add_op, x1, conv_node);
    INPUT_UPDATE(add_op, x2, bias_node);
    OUTPUT_UPDATE(add_op, y, add_node);
  }
  CHECK(conv_node);

  // ONLY support relu/leaky_relu/relu6 now
  if (!act_type.empty()) {
    if (act_type == "relu") {
      auto act_node = graph->Add<ge::op::Relu>(output_name);
      auto act_op = act_node->data<ge::op::Relu>();
      act_op->set_input_x(*conv_node->data());
      INPUT_UPDATE(act_op, x, conv_node);
      OUTPUT_UPDATE(act_op, y, act_node);
    } else if (act_type == "leaky_relu") {
      auto act_node = graph->Add<ge::op::LeakyRelu>(output_name);
      auto act_op = act_node->data<ge::op::LeakyRelu>();
      act_op->set_input_x(*conv_node->data());
      act_op->set_attr_negative_slope(leaky_relu_alpha);
      INPUT_UPDATE(act_op, x, conv_node);
      OUTPUT_UPDATE(act_op, y, act_node);
    } else if (act_type == "relu6") {
      auto act_node = graph->Add<ge::op::Relu6>(output_name);
      auto act_op = act_node->data<ge::op::Relu6>();
      act_op->set_input_x(*conv_node->data());
      INPUT_UPDATE(act_op, x, conv_node);
      OUTPUT_UPDATE(act_op, y, act_node);
    } else {
      LOG(WARNING) << "[HUAWEI_ASCEND_NPU] act type not supported: "
                   << act_type;
      return FAILED;
    }
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    deformable_conv,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::DeformableConvConverter);
