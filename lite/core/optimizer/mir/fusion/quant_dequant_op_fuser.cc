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

#include "lite/core/optimizer/mir/fusion/quant_dequant_op_fuser.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <vector>

#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

static std::string GetWeightArgname(const std::string& op_type) {
  std::string weight_argname{};
  std::vector<std::string> conv_ops = {
      "conv2d", "depthwise_conv2d", "conv2d_transpose"};
  std::vector<std::string> mul_ops = {"mul", "matmul"};
  if (std::find(conv_ops.begin(), conv_ops.end(), op_type) != conv_ops.end()) {
    weight_argname = "Filter";
  } else if (std::find(mul_ops.begin(), mul_ops.end(), op_type) !=
             mul_ops.end()) {
    weight_argname = "Y";
  }
  return weight_argname;
}

static float FindAbsMax(const float* input, int size) {
  auto abs_compare_func = [](float a, float b) {
    return (std::abs(a) < std::abs(b));
  };
  float abs_max_value =
      std::abs(*std::max_element(input, input + size, abs_compare_func));
  return abs_max_value;
}

// Per-layer quantize tensor
template <typename T>
void QuantizeTensorInPlace(Tensor* input, float scale) {
  if (input->precision() != PRECISION(kFloat)) {
    LOG(FATAL) << "Error: the precision of input should be float.";
  }
  Tensor temp_tensor;
  temp_tensor.CopyDataFrom(*input);
  input->clear();

  float* temp_data = temp_tensor.mutable_data<float>();
  T* quantized_data = input->mutable_data<T>();
  for (size_t i = 0; i < input->numel(); i++) {
    quantized_data[i] = static_cast<T>(std::round(temp_data[i] / scale));
  }
}

// Per-channel quantize tensor
template <typename T>
void QuantizeTensorInPlace(Tensor* input,
                           const std::vector<float>& scales,
                           int quant_axis) {
  if (input->precision() != PRECISION(kFloat)) {
    LOG(FATAL) << "Error: the precision of input should be float.";
  }
  if (quant_axis != 0 && quant_axis != 1) {
    LOG(FATAL) << "Input error: quant_axis should be 0 or 1.";
  }
  Tensor origin_tensor;
  origin_tensor.CopyDataFrom(*input);
  input->clear();

  auto dims = origin_tensor.dims();
  const int64_t channel = dims[quant_axis];
  if (dims.size() < 2) {
    LOG(FATAL) << "Error: the rank of input tensor should at least be 2.";
  }
  if (scales.size() != channel) {
    LOG(FATAL) << "Params Error: scale size should be equal to channel.";
  }
  float* origin_data = origin_tensor.mutable_data<float>();
  T* quantized_data = input->mutable_data<T>();

  if (quant_axis == 0) {
    const int64_t step = origin_tensor.numel() / channel;
    for (int i = 0; i < channel; i++) {
      float scale = scales[i];
      auto* src_start = origin_data + i * step;
      auto* src_end = origin_data + (i + 1) * step;
      auto* dest_start = quantized_data + i * step;
      std::transform(src_start, src_end, dest_start, [&scale](float x) {
        return static_cast<T>(std::round(x / scale));
      });
    }
  } else if (quant_axis == 1) {
    const int64_t step_i = origin_tensor.numel() / dims[0];
    const int64_t step_j = origin_tensor.numel() / (dims[0] * dims[1]);
    for (int i = 0; i < dims[0]; i++) {
      for (int j = 0; j < dims[1]; j++) {
        float scale = scales[j];
        auto* src_start = origin_data + i * step_i + j * step_j;
        auto* src_end = origin_data + i * step_i + (j + 1) * step_j;
        auto* dest_start = quantized_data + i * step_i + j * step_j;
        std::transform(src_start, src_end, dest_start, [&scale](float x) {
          return static_cast<T>(std::round(x / scale));
        });
      }
    }
  }
}

void DeleteQuantOpFuser::BuildPattern() {
  auto* input_scale_node = VarNode("input_scale_node")
                               ->assert_is_op_input(quant_op_type_, "InScale");
  auto* input_act_node =
      VarNode("input_act_node")->assert_is_op_input(quant_op_type_, "X");
  auto* quant_node =
      OpNode("quant_node", quant_op_type_)->assert_is_op(quant_op_type_);
  auto* output_scale_node =
      VarNode("output_scale_node")
          ->assert_is_op_output(quant_op_type_, "OutScale");
  auto* output_act_node =
      VarNode("output_act_node")->assert_is_op_output(quant_op_type_, "Out");

  quant_node->LinksFrom({input_scale_node, input_act_node});
  output_scale_node->LinksFrom({quant_node});
  output_act_node->LinksFrom({quant_node});
  VLOG(4) << "DeleteQuantOpFuser BuildPattern quant_op_type:" << quant_op_type_;
}

void DeleteQuantOpFuser::InsertNewNode(SSAGraph* graph,
                                       const key2nodes_t& matched) {
  auto* input_scale_node = matched.at("input_scale_node");
  auto* input_act_node = matched.at("input_act_node");
  auto* quant_node = matched.at("quant_node");
  auto* output_scale_node = matched.at("output_scale_node");
  auto* output_act_node = matched.at("output_act_node");

  // obtain scale, save attrs and relink node
  int bit_length = quant_node->stmt()->op_info()->GetAttr<int>("bit_length");
  int range = ((1 << (bit_length - 1)) - 1);
  auto* scope = quant_node->stmt()->op()->scope();
  auto* scale_tensor = scope->FindVar(output_scale_node->arg()->name)
                           ->GetMutable<lite::Tensor>();
  float scale_value = scale_tensor->data<float>()[0] / range;

  auto in_act_name = input_act_node->arg()->name;
  auto out_act_name = output_act_node->arg()->name;
  auto outlinks = output_act_node->outlinks;
  for (auto* quantized_node : outlinks) {
    // save input scale in quantized op by input argname + index
    auto op_desc = *quantized_node->stmt()->mutable_op_info();
    op_desc.SetInputScale(out_act_name, {scale_value});
    op_desc.SetAttr<int>("bit_length", bit_length);
    op_desc.UpdateAllInputs(out_act_name, in_act_name);

#ifdef LITE_WITH_FPGA
    op_desc.SetAttr("fpga_static_quant", true);
#endif

    quantized_node->stmt()->ResetOp(op_desc, graph->valid_places());
    IR_NODE_LINK_TO(input_act_node, quantized_node)
  }

  // delete nodes and edges
  std::set<const Node*> nodes2rm = {
      input_scale_node, quant_node, output_scale_node, output_act_node};
  GraphSafeRemoveNodes(graph, nodes2rm);
}

void DequantOpFuser::BuildPattern() {
  std::string weight_argname = GetWeightArgname(quantized_op_type_);
  auto* quantized_op_input = VarNode("quantized_op_input")
                                 ->assert_is_op_input(quantized_op_type_)
                                 ->AsInput();
  auto* quantized_op_weight =
      VarNode("quantized_op_weight")
          ->assert_is_op_input(quantized_op_type_, weight_argname)
          ->AsInput();
  auto* quantized_op = OpNode("quantized_op", quantized_op_type_)
                           ->assert_is_op(quantized_op_type_)
                           ->AsIntermediate();
  auto* quantized_op_out =
      VarNode("quantized_op_out")
          ->assert_is_op_output(quantized_op_type_)
          ->assert_is_op_input("fake_dequantize_max_abs", "X")
          ->AsIntermediate();
  auto* dequant_op = OpNode("dequant_op", "fake_dequantize_max_abs")
                         ->assert_is_op("fake_dequantize_max_abs")
                         ->AsIntermediate();
  auto* dequant_op_out =
      VarNode("dequant_op_out")
          ->assert_is_op_output("fake_dequantize_max_abs", "Out")
          ->AsOutput();

  quantized_op->LinksFrom({quantized_op_input, quantized_op_weight});
  quantized_op_out->LinksFrom({quantized_op});
  dequant_op->LinksFrom({quantized_op_out});
  dequant_op_out->LinksFrom({dequant_op});

  VLOG(4) << "DeQuantOpFuser BuildPattern op_type:" << quantized_op_type_;
}

void DequantOpFuser::InsertNewNode(SSAGraph* graph,
                                   const key2nodes_t& matched) {
  auto* quantized_op_input = matched.at("quantized_op_input");
  auto* quantized_op_weight = matched.at("quantized_op_weight");
  auto* quantized_op = matched.at("quantized_op");
  auto* dequant_op = matched.at("dequant_op");
  auto* dequant_op_out = matched.at("dequant_op_out");
  auto weight_name = quantized_op_weight->arg()->name;

  // obtain weight_scale from max_range
  auto* scope = quantized_op->stmt()->op()->scope();
  auto& valid_places = quantized_op->stmt()->op()->valid_places();
  int bit_length = quantized_op->stmt()->op_info()->GetAttr<int>("bit_length");
  int range = ((1 << (bit_length - 1)) - 1);
  float max_range = dequant_op->stmt()->op_info()->GetAttr<float>("max_range");
  float whole_weight_scale =
      static_cast<float>(range * range) / max_range / range;
  // As: max_range = range * range / max(abs(weight))
  // So: whole_weight_scale
  //        = range * range / (range * range / max(abs(weight))) / range
  //        = max(abs(weight)) / range

  // set op desc
  auto op_desc = *quantized_op->stmt()->op_info();
  auto quantized_weight_var_name = quantized_op_weight->arg()->name;
  auto quantized_weight_t =
      scope->FindVar(quantized_weight_var_name)->GetMutable<lite::Tensor>();

  std::vector<float> weight_scale;
  int weight_scale_size = 0;
  if (quantized_op_type_ == "conv2d" ||
      quantized_op_type_ == "depthwise_conv2d") {
    op_desc.SetInput("Input", {quantized_op_input->arg()->name});
    op_desc.SetOutput("Output", {dequant_op_out->arg()->name});
    // Conv weight shape: Cout * Cin/group * kh * hw, the weight_scale_size
    // should
    // be Cout.
    weight_scale_size = quantized_weight_t->dims()[0];
  } else if (quantized_op_type_ == "conv2d_transpose") {
    op_desc.SetInput("Input", {quantized_op_input->arg()->name});
    op_desc.SetOutput("Output", {dequant_op_out->arg()->name});

    auto* conv_op_desc = matched.at("quantized_op")->stmt()->op_info();
    auto groups = conv_op_desc->GetAttr<int>("groups");
    // Conv weight shape: Cin * Cout/group * kh * hw, the weight_scale_size
    // should
    // be Cout.
    weight_scale_size = quantized_weight_t->dims()[1] * groups;
  } else if (quantized_op_type_ == "mul" || quantized_op_type_ == "matmul") {
    op_desc.SetInput("X", {quantized_op_input->arg()->name});
    op_desc.SetOutput("Out", {dequant_op_out->arg()->name});
    // Fc weight: Cin * Cout, the weight_scale_size should be Cout.
    weight_scale_size = quantized_weight_t->dims()[1];
  }
  for (int i = 0; i < weight_scale_size; i++) {
    weight_scale.push_back(whole_weight_scale);
  }
#ifndef LITE_WITH_FPGA
  op_desc.SetAttr("enable_int8", true);
#endif

  op_desc.SetInputScale(weight_name, weight_scale);

  // change the weight from the float type to int8 type.
  Tensor temp_tensor;
  temp_tensor.CopyDataFrom(*quantized_weight_t);
  float* temp_data = temp_tensor.mutable_data<float>();
  size_t weight_num = quantized_weight_t->data_size();

#ifdef LITE_WITH_FPGA
  float* quantized_weight_data = quantized_weight_t->mutable_data<float>();
  for (size_t i = 0; i < weight_num; i++) {
    quantized_weight_data[i] = temp_data[i] * whole_weight_scale;
  }
  quantized_weight_t->set_persistable(true);
  quantized_weight_t->set_precision(PRECISION(kFloat));
#else
  int8_t* quantized_weight_data = quantized_weight_t->mutable_data<int8_t>();
  for (size_t i = 0; i < weight_num; i++) {
    quantized_weight_data[i] = static_cast<int8_t>(temp_data[i]);
  }
  quantized_weight_t->set_persistable(true);
  quantized_weight_t->set_precision(PRECISION(kInt8));
#endif

  // new op and relink nodes
  auto new_quantized_op = LiteOpRegistry::Global().Create(quantized_op_type_);
  new_quantized_op->Attach(op_desc, scope);
  auto* new_quantized_op_node =
      graph->GraphCreateInstructNode(new_quantized_op, valid_places);
  IR_NODE_LINK_TO(quantized_op_input, new_quantized_op_node);
  IR_NODE_LINK_TO(quantized_op_weight, new_quantized_op_node);
  IR_NODE_LINK_TO(new_quantized_op_node, dequant_op_out);
}

void ChannelWiseDequantOpFuser::BuildPattern() {
  std::string dequant_op_type = "fake_channel_wise_dequantize_max_abs";
  std::string weight_argname = GetWeightArgname(quantized_op_type_);
  auto* quantized_op_input = VarNode("quantized_op_input")
                                 ->assert_is_op_input(quantized_op_type_)
                                 ->AsInput();
  auto* quantized_op_weight =
      VarNode("quantized_op_weight")
          ->assert_is_op_input(quantized_op_type_, weight_argname)
          ->AsInput();
  auto* quantized_op = OpNode("quantized_op", quantized_op_type_)
                           ->assert_is_op(quantized_op_type_)
                           ->AsIntermediate();
  auto* quantized_op_out = VarNode("quantized_op_out")
                               ->assert_is_op_output(quantized_op_type_)
                               ->assert_is_op_input(dequant_op_type, "X")
                               ->AsIntermediate();
  // The scale var_node of input activation is deleted in DeleteQuantOpFuser
  auto* dequant_op_channel_scale = VarNode("dequant_op_channel_scale")
                                       ->assert_is_op_input(dequant_op_type)
                                       ->AsIntermediate();
  auto* dequant_op = OpNode("dequant_op", dequant_op_type)
                         ->assert_is_op(dequant_op_type)
                         ->AsIntermediate();
  auto* dequant_op_out = VarNode("dequant_op_out")
                             ->assert_is_op_output(dequant_op_type, "Out")
                             ->AsOutput();

  quantized_op->LinksFrom({quantized_op_input, quantized_op_weight});
  quantized_op_out->LinksFrom({quantized_op});
  dequant_op->LinksFrom({quantized_op_out, dequant_op_channel_scale});
  dequant_op_out->LinksFrom({dequant_op});

  VLOG(4) << "ChannelWiseDequantOpFuser BuildPattern op_type:"
          << quantized_op_type_;
}

void ChannelWiseDequantOpFuser::InsertNewNode(SSAGraph* graph,
                                              const key2nodes_t& matched) {
  auto* quantized_op_input = matched.at("quantized_op_input");
  auto* quantized_op_weight = matched.at("quantized_op_weight");
  auto* quantized_op = matched.at("quantized_op");
  auto* dequant_op_channel_scale = matched.at("dequant_op_channel_scale");
  auto* dequant_op = matched.at("dequant_op");
  auto* dequant_op_out = matched.at("dequant_op_out");
  auto weight_name = quantized_op_weight->arg()->name;

  // obtain input weight_scale from fake_dequant op
  auto* scope = quantized_op->stmt()->op()->scope();
  auto& valid_places = quantized_op->stmt()->op()->valid_places();

  std::vector<float> weight_scale;
  std::vector<int> quant_bits =
      dequant_op->stmt()->op_info()->GetAttr<std::vector<int>>("quant_bits");
  int weight_bit_length = quant_bits[0];
  int range = ((1 << (weight_bit_length - 1)) - 1);
  auto channel_scale_name = dequant_op_channel_scale->arg()->name;
  auto channel_scale_tensor =
      scope->FindVar(channel_scale_name)->GetMutable<lite::Tensor>();
  auto* channel_scale_data = channel_scale_tensor->data<float>();
  for (size_t i = 0; i < channel_scale_tensor->data_size(); i++) {
    weight_scale.push_back(channel_scale_data[i] / range);
  }

  // set op desc
  auto op_desc = *quantized_op->stmt()->op_info();
  if (quantized_op_type_ == "conv2d" ||
      quantized_op_type_ == "depthwise_conv2d" ||
      quantized_op_type_ == "conv2d_transpose") {
    op_desc.SetInput("Input", {quantized_op_input->arg()->name});
    op_desc.SetOutput("Output", {dequant_op_out->arg()->name});
  } else if (quantized_op_type_ == "mul" || quantized_op_type_ == "matmul") {
    op_desc.SetInput("X", {quantized_op_input->arg()->name});
    op_desc.SetOutput("Out", {dequant_op_out->arg()->name});
  }

#ifndef LITE_WITH_FPGA
  op_desc.SetAttr("enable_int8", true);
#endif
  op_desc.SetInputScale(weight_name, weight_scale);

  // change the weight from the float type to int8 type.
  auto quantized_weight_var_name = quantized_op_weight->arg()->name;
  auto quantized_weight_t =
      scope->FindVar(quantized_weight_var_name)->GetMutable<lite::Tensor>();
  Tensor temp_tensor;
  temp_tensor.CopyDataFrom(*quantized_weight_t);
  float* temp_data = temp_tensor.mutable_data<float>();

#ifdef LITE_WITH_FPGA
  float* quantized_weight_data = quantized_weight_t->mutable_data<float>();
  int channel = channel_scale_tensor->data_size();
  int weight_chw = quantized_weight_t->data_size() / channel;
  for (size_t i = 0; i < quantized_weight_t->data_size(); i++) {
    int c = i / weight_chw;
    quantized_weight_data[i] = temp_data[i] * weight_scale[c];
  }
  quantized_weight_t->set_persistable(true);
  quantized_weight_t->set_precision(PRECISION(kFloat));

#else
  int8_t* quantized_weight_data = quantized_weight_t->mutable_data<int8_t>();
  for (size_t i = 0; i < quantized_weight_t->data_size(); i++) {
    quantized_weight_data[i] = static_cast<int8_t>(temp_data[i]);
  }
  quantized_weight_t->set_persistable(true);
  quantized_weight_t->set_precision(PRECISION(kInt8));
#endif

  // new op and relink nodes
  auto new_quantized_op = LiteOpRegistry::Global().Create(quantized_op_type_);
  new_quantized_op->Attach(op_desc, scope);
  auto* new_quantized_op_node =
      graph->GraphCreateInstructNode(new_quantized_op, valid_places);
  IR_NODE_LINK_TO(quantized_op_input, new_quantized_op_node);
  IR_NODE_LINK_TO(quantized_op_weight, new_quantized_op_node);
  IR_NODE_LINK_TO(new_quantized_op_node, dequant_op_out);
}

void QuantDequantOpFuser::BuildPattern() {
  auto* input_var_node = VarNode("input_var_node")
                             ->assert_is_op_input(quant_dequant_op_type_, "X");
  auto* quant_dequant_node =
      OpNode("quant_dequant_node", quant_dequant_op_type_)
          ->assert_is_op(quant_dequant_op_type_);
  auto* output_scale_node =
      VarNode("output_scale_node")
          ->assert_is_op_output(quant_dequant_op_type_, "OutScale");
  auto* output_var_node =
      VarNode("output_var_node")
          ->assert_is_op_output(quant_dequant_op_type_, "Out");

  if (quant_dequant_op_type_ ==
      "fake_quantize_dequantize_moving_average_abs_max") {
    auto* input_scale_node =
        VarNode("input_scale_node")
            ->assert_is_op_input(quant_dequant_op_type_, "InScale");
    quant_dequant_node->LinksFrom({input_scale_node, input_var_node});
  } else {
    quant_dequant_node->LinksFrom({input_var_node});
  }
  output_scale_node->LinksFrom({quant_dequant_node});
  output_var_node->LinksFrom({quant_dequant_node});
}

void QuantDequantOpFuser::InsertNewNode(SSAGraph* graph,
                                        const key2nodes_t& matched) {
  auto* input_var_node = matched.at("input_var_node");
  auto* quant_dequant_node = matched.at("quant_dequant_node");
  auto* output_scale_node = matched.at("output_scale_node");
  auto* output_var_node = matched.at("output_var_node");

  auto input_var_name = input_var_node->arg()->name;
  auto output_var_name = output_var_node->arg()->name;
  bool input_var_is_activation = !input_var_node->arg()->is_weight;

  // 1. Get thresholds and scales
  // The activation only has a scale.
  // When the weight is per-layer quantized, it has a scale.
  // When the weight is per-channel quantized, the num of scales is equal
  // to the output channel of the weight.
  auto* scope = quant_dequant_node->stmt()->op()->scope();
  auto* input_var_tensor =
      scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();

  std::vector<float> thresholds;
  if (input_var_is_activation) {
    CHECK_EQ(quant_dequant_op_type_,
             "fake_quantize_dequantize_moving_average_abs_max")
        << "The quant_dequant type of activation should be "
        << "fake_quantize_dequantize_moving_average_abs_max for now.";
    auto* scale_tensor = scope->FindVar(output_scale_node->arg()->name)
                             ->GetMutable<lite::Tensor>();
    thresholds.push_back(scale_tensor->data<float>()[0]);
  } else {
    CHECK(quant_dequant_op_type_ == "fake_quantize_dequantize_abs_max" ||
          quant_dequant_op_type_ ==
              "fake_channel_wise_quantize_dequantize_abs_max")
        << "The quant_dequant type of weight should be "
        << "fake_quantize_dequantize_abs_max or "
        << "fake_channel_wise_quantize_dequantize_abs_max.";
    if (quant_dequant_op_type_ == "fake_quantize_dequantize_abs_max") {
      auto* input_var_data = input_var_tensor->data<float>();
      float threshold = FindAbsMax(input_var_data, input_var_tensor->numel());
      thresholds.push_back(threshold);
    } else {
      auto* scale_tensor = scope->FindVar(output_scale_node->arg()->name)
                               ->GetMutable<lite::Tensor>();
      int64_t num = scale_tensor->numel();
      thresholds.resize(num);
      memcpy(
          thresholds.data(), scale_tensor->data<float>(), num * sizeof(float));
    }
  }

  int bit_length =
      quant_dequant_node->stmt()->op_info()->GetAttr<int>("bit_length");
  std::vector<float> scales(thresholds.size(), 0);
  std::transform(
      thresholds.begin(),
      thresholds.end(),
      scales.begin(),
      [&bit_length](float x) { return x / ((1 << (bit_length - 1)) - 1); });

  // 2. Update op_info of the quantized op
  for (auto* quantized_node : output_var_node->outlinks) {
    auto op_info = *quantized_node->stmt()->op_info();
    op_info.UpdateAllInputs(output_var_name, input_var_name);
    op_info.SetAttr<int>("bit_length", bit_length);

    if (input_var_is_activation) {
      op_info.SetInputScale(input_var_name, scales);
    } else {
      std::string op_type = op_info.Type();
      const int quant_axis =
          (op_type == "conv2d" || op_type == "depthwise_conv2d") ? 0 : 1;
      if (quant_dequant_op_type_ == "fake_quantize_dequantize_abs_max") {
        // Scales of weights are vector, so expand the scale to vector
        // the quant axis of conv2d and depthwise_conv2d is 0
        // the quant axis of conv2d_transpose (consider group), mul and matmul
        // is 1
        int scale_size = input_var_tensor->dims()[quant_axis];
        op_info.SetInputScale(input_var_name,
                              std::vector<float>(scale_size, scales.front()));
      } else {
        op_info.SetInputScale(input_var_name, scales);
      }
      // PaddleLite only supports this int8 ops for now
      // TODO(pjc) : support conv2d_transpose
      if (op_type == "mul" || op_type == "matmul" || op_type == "conv2d" ||
          op_type == "depthwise_conv2d" || op_type == "conv2d_transpose") {
#ifndef LITE_WITH_FPGA
        op_info.SetAttr("enable_int8", true);
#endif
        if (scales.size() == 1) {
          QuantizeTensorInPlace<int8_t>(input_var_tensor, scales.front());
        } else {
          QuantizeTensorInPlace<int8_t>(input_var_tensor, scales, quant_axis);
        }
      }
    }

    quantized_node->stmt()->ResetOp(op_info, graph->valid_places());
    IR_NODE_LINK_TO(input_var_node, quantized_node);
  }

  // 3. Delete nodes and edges
  std::set<const Node*> nodes2rm = {
      quant_dequant_node, output_scale_node, output_var_node};
  if (quant_dequant_op_type_ ==
      "fake_quantize_dequantize_moving_average_abs_max") {
    auto* input_scale_node = matched.at("input_scale_node");
    nodes2rm.insert(input_scale_node);
  }
  GraphSafeRemoveNodes(graph, nodes2rm);
}

void DynamicQuantOpFuser::BuildPattern() {
  auto* weight_node =
      VarNode("weight_node")->assert_is_op_input(op_type_, input_argname_);
  // op_node should have "quantization_type" attribute
  auto* op_node =
      OpNode("op_node", op_type_)
          ->assert_is_op(op_type_)
          ->assert_op_attr_satisfied<std::string>(
              "quantization_type", [](const std::string& x) { return true; });
  op_node->LinksFrom({weight_node});
}

void DynamicQuantOpFuser::InsertNewNode(SSAGraph* graph,
                                        const key2nodes_t& matched) {
  auto* op_node = matched.at("op_node");
  auto* weight_node = matched.at("weight_node");

  auto* scope = op_node->stmt()->op()->scope();
  std::string weight_name = weight_node->arg()->name;
  auto weight_tensor = scope->FindVar(weight_name)->GetMutable<lite::Tensor>();
  auto weight_dims = weight_tensor->dims();
  CHECK(weight_dims.size() == 2) << "The rank of weight should be 2.";
  VLOG(4) << "Quantizes weight of lstm or gru:" << weight_name;

  // process weight scale
  auto op_info = *op_node->stmt()->mutable_op_info();
  auto bit_length = op_info.GetAttr<int>("bit_length");
  auto weight_threshold =
      op_info.GetAttr<float>(input_argname_ + "0_threshold");
  float weight_scale = weight_threshold / ((1 << (bit_length - 1)) - 1);
  std::vector<float> weight_scale_vct(weight_dims[1], weight_scale);

  op_info.SetAttr("enable_int8", true);
  op_info.SetAttr("bit_length", bit_length);
  op_info.SetInputScale(weight_name, weight_scale_vct);
  op_node->stmt()->ResetOp(op_info, graph->valid_places());

  // convert the weight from float to int8
  Tensor temp_tensor;
  temp_tensor.CopyDataFrom(*weight_tensor);
  weight_tensor->clear();

  auto* temp_data = temp_tensor.data<float>();
  auto* weight_data = weight_tensor->mutable_data<int8_t>();
  int64_t weight_num = weight_tensor->data_size();
  for (size_t i = 0; i < weight_num; i++) {
    weight_data[i] =
        static_cast<int8_t>(std::round(temp_data[i] / weight_scale));
  }
  weight_tensor->set_persistable(true);
  weight_tensor->set_precision(PRECISION(kInt8));
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
