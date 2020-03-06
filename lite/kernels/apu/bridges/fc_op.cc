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

#include "lite/kernels/apu/bridges/graph.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/apu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

int FCConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto model = graph->model();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[APU] Converting [" + op_type + "]";

  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  CHECK_GE(input_dims.size(), 2UL);
  auto w_name = op_info->Input("W").front();
  auto w = scope->FindMutableTensor(w_name);
  auto w_dims = w->dims();
  CHECK_EQ(w_dims.size(), 2UL);
  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  int m = input_dims.Slice(0, in_num_col_dims).production();
  int k = input_dims.Slice(in_num_col_dims, input_dims.size()).production();
  int n = w_dims[1];
  CHECK_EQ(k * n, w_dims.production());
  VLOG(3) << "[APU] input dims: " << input_dims << " w dims: " << w_dims << " out_dims: " << out_dims
          << " m: " << m << " k: " << k << " n: " << n;

  float input_scale = 1.0f;
  float out_scale = 1.0f;
  std::vector<float> w_scale;
  if (op_info->HasAttr("enable_int8")) {
    if (op_info->GetAttr<bool>("enable_int8")) {
      if (op_info->HasAttr("input_scale"))
        input_scale = op_info->GetAttr<float>("input_scale");
      if (op_info->HasAttr("weight_scale"))
        w_scale = op_info->GetAttr<std::vector<float>>("weight_scale");
      if (op_info->HasAttr("output_scale"))
        out_scale = op_info->GetAttr<float>("output_scale");
    } else {
      return FAILED;
    }
  } else {
    return FAILED;
  }

  for(int i = 0; i < w_scale.size(); i++)
     VLOG(3) << w_scale[0];

  // Add input tensor type
  NeuronOperandType inType;
  inType.type = NEURON_TENSOR_QUANT8_ASYMM;
  inType.scale = input_scale;
  inType.zeroPoint = 0;
  inType.dimensionCount = input_dims.size();
  std::vector<uint32_t>dims_in = {(uint32_t)input_dims[0], (uint32_t)input_dims[2],
                                  (uint32_t)input_dims[3], (uint32_t)input_dims[1]};
  /* Use this after NHWC
  for (int i = 0; i < input_dims.size(); i++)
    dims_in.push_back(input_dims[i]);
  */
  inType.dimensions = &dims_in[0];
  std::shared_ptr<Node> in_node = nullptr;
  if (graph->Has(input_name)) {
    // input operand already exist
    in_node = graph->Get(input_name);
    LOG(INFO) << "Graph has " << input_name << ",index: " << in_node->index();
  } else {
    // add input operand
    NeuronModel_addOperand(model, &inType); //input
    in_node = graph->Add(input_name, dims_in);
  }
  VLOG(3) << "input_scale: " << input_scale << ", inType: " << inType.dimensions[0]
          << " : " << inType.dimensions[1] << " : "
          << inType.dimensions[2] << " : "  << inType.dimensions[3];

  NeuronOperandType wType;
  wType.type = NEURON_TENSOR_QUANT8_ASYMM;
  wType.scale = w_scale[0];
  wType.zeroPoint = 0;
  wType.dimensionCount = w_dims.size();
  std::vector<uint32_t>dims_w = {(uint32_t)w_dims[1], (uint32_t)w_dims[0]};
  wType.dimensions = &dims_w[0];
  NeuronModel_addOperand(model, &wType); //1: weight
  std::shared_ptr<Node> w_node = nullptr;
  w_node = graph->Add(w_name, dims_w);
  VLOG(3) << "w_scale size: " << w_scale.size() << ",w_scale: " << w_scale[0] << ", wType dimensions: " << wType.dimensions[0]
          << " : " << wType.dimensions[1] << ", memory size: " << w->memory_size();

  // Add bias type
  NeuronOperandType biasType;
  biasType.type = NEURON_TENSOR_INT32;
  biasType.zeroPoint = 0;
  biasType.scale = input_scale * w_scale[0];
  std::shared_ptr<Node> bias_node = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias_type = kernel->GetInputDeclType("Bias");
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    auto bias_data_size = bias_dims.production();

    biasType.dimensionCount = bias_dims.size();
    std::vector<uint32_t> dims_bias = {(uint32_t)bias_dims[0]};
    biasType.dimensions = &dims_bias[0];
    NeuronModel_addOperand(model, &biasType); //2: bias
    bias_node = graph->Add(bias_name, dims_bias);
    VLOG(3) << "Bias name: " << bias_name << ", bias dims: " << bias_dims << ", bias scale: "<< biasType.scale << " ,memory size: " << bias->memory_size();
  } else {
    biasType.dimensionCount = 1;
    std::vector<uint32_t> dims_bias = {(uint32_t)n};
    biasType.dimensions = &dims_bias[0];
    NeuronModel_addOperand(model, &biasType); //2: bias
    bias_node = graph->Add(w_name + "_default_bias", dims_bias);
  }

  // Add fuse type
  NeuronOperandType fuseType;
  fuseType.type = NEURON_INT32;
  fuseType.dimensionCount = 0;
  std::vector<uint32_t> dims_int32 = {0};
  NeuronModel_addOperand(model, &fuseType); //3: fuse
  std::shared_ptr<Node> fuse_node = nullptr;
  fuse_node = graph->Add(w_name + "_fuse", dims_int32);

  // Add output tensor type
  NeuronOperandType outType;
  outType.type = NEURON_TENSOR_QUANT8_ASYMM;
  outType.scale = out_scale;
  outType.zeroPoint = 0;
  outType.dimensionCount = 2;
  std::vector<uint32_t>dims_out = {(uint32_t)out_dims[0], out_dims[1]};
  outType.dimensions = &dims_out[0];
  VLOG(3) << "out_scale: " << out_scale << ", outType: " << outType.dimensions[0] << " : " << outType.dimensions[1];
  NeuronModel_addOperand(model, &outType); //output
  std::shared_ptr<Node> out_node = nullptr;
  out_node = graph->Add(out_name, dims_out);

  int  neuron_errCode = NeuronModel_setOperandValue(model, w_node->index(), w->raw_data(), w->memory_size()); //1: weight
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Set W operand value fail:" << neuron_errCode << ",index: " << w_node->index();
    return FAILED;
  }

  // Add bias if bias tensor exists
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias_type = kernel->GetInputDeclType("Bias");
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    neuron_errCode = NeuronModel_setOperandValue(model, bias_node->index(), bias->raw_data(), bias->memory_size()); //2: bias
  } else {
    Tensor default_bias;
    default_bias.Resize({k});
    default_bias.mutable_data<int32_t>();
    VLOG(3) << "bais_default: " << default_bias.memory_size();
    neuron_errCode = NeuronModel_setOperandValue(model, bias_node->index(), default_bias.raw_data(), default_bias.memory_size()); //2: bias
  }

  // Add fuse value
  int32_t fuse_val[1] = {0};
  NeuronModel_setOperandValue(model, fuse_node->index(), fuse_val, sizeof(int32_t) * 1); //3: fuse

  std::vector<uint32_t>addInIndex = {in_node->index(),
                                     w_node->index(),
                                     bias_node->index(),
                                     fuse_node->index()};
  std::vector<uint32_t>addOutIndex = {out_node->index()};
  neuron_errCode = NeuronModel_addOperation(model, NEURON_FULLY_CONNECTED,
                                      addInIndex.size(),  &addInIndex[0],
                                      addOutIndex.size(), &addOutIndex[0]);

  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Add op fail:" << op_type;
    return FAILED;
  }

  VLOG(3) << "Add " << op_type << " success! \n";

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fc, kAPU, paddle::lite::subgraph::apu::FCConverter);
