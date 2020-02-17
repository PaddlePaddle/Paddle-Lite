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
#include "lite/core/mir/pattern_matcher_high_api.h"

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
*/
class DeleteQuantOpFuser : public FuseBase {
 public:
  explicit DeleteQuantOpFuser(const std::string& quant_op_type)
      : quant_op_type_(quant_op_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;

 private:
  std::string quant_op_type_{};
};
class DeleteDynamicQuantOpFuser : public FuseBase {
 public:
  explicit DeleteDynamicQuantOpFuser(const std::string& quant_op_type)
      : quant_op_type_(quant_op_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;

 private:
  std::string quant_op_type_{};
};

/* DequantOpFuser process conv2d/depthwise_conv2d/mul + fake_dequantize_max_abs.
*/
class DequantOpFuser : public FuseBase {
 public:
  explicit DequantOpFuser(const std::string& quantized_op_type)
      : quantized_op_type_(quantized_op_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;

 private:
  std::string quantized_op_type_{};
};

/* ChannelWiseDequantOpFuser process conv2d/depthwise_conv2d +
 * fake_channel_wise_dequantize_max_abs.
*/
class ChannelWiseDequantOpFuser : public FuseBase {
 public:
  explicit ChannelWiseDequantOpFuser(const std::string& quantized_op_type)
      : quantized_op_type_(quantized_op_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;

 private:
  std::string quantized_op_type_{};
};

/* The pattern like "fake_quantize_dequantize_moving_average_abs_max +
 * pooled/elementwise_add" can be deteted by this fuser. The fuser
 * extract the input_scale form fake_quant_dequant_op and save into
 * the quantized_op. Besides, the fuser delete fake_quant_dequant_op in
 * the graph.
*/

class DeleteQuantDequantOpFuser : public FuseBase {
 public:
  explicit DeleteQuantDequantOpFuser(const std::string& quantized_op_type)
      : quantized_op_type_(quantized_op_type) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;

 private:
  std::string quantized_op_type_{};
};
// dynamic quantdequant op fuser
class DynamicQuantDequantOpFuser : public FuseBase {
 public:
  explicit DynamicQuantDequantOpFuser(const std::string& quantized_op_type,
                                      const std::string& op_type,
                                      int i)
      : op_type_(op_type), quant_type_(quantized_op_type), times_(i) {}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;

 private:
  std::string op_type_{};
  std::string quant_type_{};
  int times_{1};
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
