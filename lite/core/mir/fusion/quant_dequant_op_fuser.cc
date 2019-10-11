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

#include "lite/core/mir/fusion/quant_dequant_op_fuser.h"
#include <memory>
#include <vector>
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void QuantDequantOpFuser::BuildPattern() {
  std::string weight_name = "";
  if (op_type_ == "conv2d" || op_type_ == "depthwise_conv2d") {
    weight_name = "Filter";
  } else {
    weight_name = "Y";
  }

  auto* quantized_op_input =
      VarNode("quantized_op_input")->assert_is_op_input(op_type_)->AsInput();
  auto* quantized_op_weight = VarNode("quantized_op_weight")
                                  ->assert_is_op_input(op_type_, weight_name)
                                  ->AsInput();
  auto* quantized_op = OpNode("quantized_op", op_type_)
                           ->assert_is_op(op_type_)
                           ->AsIntermediate();
  auto* quantized_op_out =
      VarNode("quantized_op_out")
          ->assert_is_op_output(op_type_)
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
}

void QuantDequantOpFuser::InsertNewNode(SSAGraph* graph,
                                        const key2nodes_t& matched) {
  auto* quant_op_input = matched.at("quantized_op_input");
  auto* quantized_op_weight = matched.at("quantized_op_weight");
  auto* quantized_op = matched.at("quantized_op");
  auto* dequant_op = matched.at("dequant_op");
  auto* dequant_op_out = matched.at("dequant_op_out");

  // obtain input_scale and weight_scale
  auto* scope = quantized_op->stmt()->op()->scope();
  auto& valid_places = quantized_op->stmt()->op()->valid_places();
  int bit_length = quantized_op->stmt()->op_info()->GetAttr<int>("bit_length");
  int range = ((1 << (bit_length - 1)) - 1);
  float input_scale =
      quantized_op->stmt()->op_info()->GetAttr<float>("input_scale");
  float max_range = dequant_op->stmt()->op_info()->GetAttr<float>("max_range");
  float whole_weight_scale =
      static_cast<float>(range * range) / max_range / range;
  // max_range = range * range / max(abs(weight))
  // weight_scale = range * range / (range * range / max(abs(weight))) / range
  //              = max(abs(weight)) / range

  // set op desc
  cpp::OpDesc op_desc = *quantized_op->stmt()->op_info();
  auto quantized_weight_var_name = quantized_op_weight->arg()->name;
  auto quantized_weight_t =
      scope->FindVar(quantized_weight_var_name)->GetMutable<lite::Tensor>();
  std::vector<float> weight_scale;
  int weight_scale_size;
  if (op_type_ == "conv2d" || op_type_ == "depthwise_conv2d") {
    op_desc.SetInput("Input", {quant_op_input->arg()->name});
    op_desc.SetOutput("Output", {dequant_op_out->arg()->name});
    // Conv weight shape: Cout * Cin * kh * hw, the weight_scale_size should
    // be Cout.
    weight_scale_size = quantized_weight_t->dims()[0];
  } else if (op_type_ == "mul") {
    op_desc.SetInput("X", {quant_op_input->arg()->name});
    op_desc.SetOutput("Out", {dequant_op_out->arg()->name});
    // Fc weight: Cin * Cout, the weight_scale_size should be Cout.
    weight_scale_size = quantized_weight_t->dims()[1];
  }
  for (int i = 0; i < weight_scale_size; i++) {
    weight_scale.push_back(whole_weight_scale);
  }
  op_desc.SetAttr("enable_int8", true);
  op_desc.SetAttr("input_scale", input_scale);
  op_desc.SetAttr("weight_scale", weight_scale);

  // change the weight from the float type to int8 type.
  Tensor temp_tensor;
  temp_tensor.CopyDataFrom(*quantized_weight_t);
  float* temp_data = temp_tensor.mutable_data<float>();
  size_t weight_num = quantized_weight_t->data_size();
  int8_t* quantized_weight_data = quantized_weight_t->mutable_data<int8_t>();
  for (size_t i = 0; i < weight_num; i++) {
    quantized_weight_data[i] = static_cast<int8_t>(temp_data[i]);
  }
  quantized_weight_t->set_persistable(true);
  quantized_weight_t->set_precision(PRECISION(kInt8));

  // new op and relink nodes
  auto new_quantized_op = LiteOpRegistry::Global().Create(op_type_);
  new_quantized_op->Attach(op_desc, scope);
  auto* new_quantized_op_node =
      graph->GraphCreateInstructNode(new_quantized_op, valid_places);
  IR_NODE_LINK_TO(quant_op_input, new_quantized_op_node);
  IR_NODE_LINK_TO(quantized_op_weight, new_quantized_op_node);
  IR_NODE_LINK_TO(new_quantized_op_node, dequant_op_out);
}

cpp::OpDesc QuantDequantOpFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc;
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
