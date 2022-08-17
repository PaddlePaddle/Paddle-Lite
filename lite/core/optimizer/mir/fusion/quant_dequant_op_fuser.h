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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/* The model trained by fluid quantization is a simulation of real int8.
 * The quantized Ops(conv2d, mul, depthwise conv2d etc) have fake_quant op
 * in front and fake_dequant op behind.
 *
 * For int8 mode, the pattern like "fake_quant + quantized_op + fake_dequant"
 * can be processed by the following three fuser. The fuser extract the
 * input_scale and the weight_scale info from fake_quant, fake_dequant op and
 * fuse those into the quantized_op.
 * In addition, the fuser delete fake_quant and fake_dequant op in the graph at
 * the last.
*/

/* DeleteQuantOpFuser process
 * fake_quantize_range_abs_max/fake_quantize_moving_average_abs_max
 * + conv2d/mul/depthwise.
 *
 * 1. Set next op's input scale info.
 * 2. Delete quant op
*/
class DeleteQuantOpFuser : public FuseBase {
 public:
  explicit DeleteQuantOpFuser(const std::string& quant_op_type)
      : quant_op_type_(quant_op_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  std::string quant_op_type_{};
};

/* DequantOpFuser process conv2d/depthwise_conv2d/mul + fake_dequantize_max_abs.
 *
 * 1. Set previous op's weight scale info.
 * 2. Restore float32 weight to int8.
 * 3. Delete fake_dequantize_max_abs op.
*/
class DequantOpFuser : public FuseBase {
 public:
  explicit DequantOpFuser(const std::string& quantized_op_type)
      : quantized_op_type_(quantized_op_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  std::string quantized_op_type_{};
};

/* ChannelWiseDequantOpFuser process conv2d/depthwise_conv2d +
 * fake_channel_wise_dequantize_max_abs.
 *
 * 1. Set previous op's weight channel wise scale info.
 * 2. Cast previous op's weight to int8.
 * 3. Delete dequant op.
*/
class ChannelWiseDequantOpFuser : public FuseBase {
 public:
  explicit ChannelWiseDequantOpFuser(const std::string& quantized_op_type)
      : quantized_op_type_(quantized_op_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  std::string quantized_op_type_{};
};

/* The pattern like "input_var + fake_quantize_dequantize_op +
 * quant_dequant_var + quantized_op" can be deteted by this fuser.
 * The fuser sets the input scale for the quantized_op and
 * deletes the fake_quant_dequant_op. If the input_var is weight,
 * The fuser also quantizes the input_var.
 *
 * 1. Set next op's input scale info.
 * 2. Delete quant_dequant_op
*/
class QuantDequantOpFuser : public FuseBase {
 public:
  explicit QuantDequantOpFuser(const std::string& quant_dequant_op_type)
      : quant_dequant_op_type_(quant_dequant_op_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  std::string quant_dequant_op_type_{};
  std::vector<std::string> input_activation_quant_op = {
      "matmul_v2", "matmul", "mul"};
};

/* DynamicQuantOpFuser is applied for LSTM and GRU for now.
 * This fuser collects the weight scale and convert the weight from fp32
 * to int8.
*/

class DynamicQuantOpFuser : public FuseBase {
 public:
  explicit DynamicQuantOpFuser(const std::string& op_type,
                               const std::string& input_argname) {
    op_type_ = op_type;
    input_argname_ = input_argname;
  }

  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  std::string op_type_{};
  std::string input_argname_{};
};

/* The pattern like "input_var + quantize_linear_op + dequantize_linear_op +
 * quantized_op " can be deteted by this fuser.
 * The fuser sets the input scale for the quantized_op and
 * deletes the "quantize_linear_op + dequantize_linear_op".
 * If the input_var is weight, the fuser also quantizes the input_var.
*/
class QuantDequantLinearOpFuser : public FuseBase {
 public:
  QuantDequantLinearOpFuser() {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  std::vector<std::string> quant_op_types_ = {"conv2d",
                                              "depthwise_conv2d",
                                              "conv2d_transpose",
                                              "depthwise_conv2d_transpose",
                                              "mul",
                                              "matmul",
                                              "matmul_v2"};
};

/* The pattern like "dequantize_linear_op + quantized_op "
 *  can be deteted by this fuser.
 * The fuser sets the weight scale for the quantized_op and
 * deletes the "dequantize_linear_op".
*/
class DequantLinearOpFuser : public FuseBase {
 public:
  DequantLinearOpFuser() {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  std::string dequant_op_type_{};
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
