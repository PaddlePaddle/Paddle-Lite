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
#include "lite/kernels/apu/bridges/graph.h"
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

  CHECK(op_info->HasAttr("enable_int8") &&
        op_info->GetAttr<bool>("enable_int8"));

  // Get input and output vars and op attributes
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
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  int m = input_dims.Slice(0, in_num_col_dims).production();
  int k = input_dims.Slice(in_num_col_dims, input_dims.size()).production();
  int n = w_dims[1];
  CHECK_EQ(k * n, w_dims.production());
  VLOG(3) << "[APU] input dims: " << input_dims << " w dims: " << w_dims
          << " out_dims: " << out_dims << " m: " << m << " k: " << k
          << " n: " << n;

  CHECK(op_info->HasInputScale(input_scale_name, true));
  auto input_scale = op_info->GetInputScale(input_scale_name, true)[0];
  CHECK(op_info->HasInputScale(w_scale_name, true));
  auto w_scale = op_info->GetInputScale(w_scale_name, true);
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  auto out_scale = op_info->GetOutputScale(out_scale_name, true)[0];

  // Add input tensor type
  NeuronOperandType inType;
  inType.type = NEURON_TENSOR_QUANT8_ASYMM;
  inType.scale = input_scale;
  inType.zeroPoint = 128;
  inType.dimensionCount = input_dims.size();
  std::vector<uint32_t> dims_in = {(uint32_t)input_dims[0],
                                   (uint32_t)input_dims[2],
                                   (uint32_t)input_dims[3],
                                   (uint32_t)input_dims[1]};

  inType.dimensions = &dims_in[0];
  std::shared_ptr<Node> in_node = nullptr;
  if (graph->Has(input_name)) {
    in_node = graph->Get(input_name);
    VLOG(3) << "Graph has " << input_name << ",index: " << in_node->index();
  } else {
    NeuronModel_addOperand(model, &inType);  // Operand 0: input
    in_node = graph->Add(input_name, dims_in);
  }
  VLOG(3) << "input_scale: " << input_scale
          << ", inType: " << inType.dimensions[0] << " : "
          << inType.dimensions[1] << " : " << inType.dimensions[2] << " : "
          << inType.dimensions[3];

  NeuronOperandType wType;
  wType.type = NEURON_TENSOR_QUANT8_ASYMM;
  wType.scale = w_scale[0];
  wType.zeroPoint = 128;
  wType.dimensionCount = w_dims.size();
  std::vector<uint32_t> dims_w = {(uint32_t)w_dims[1], (uint32_t)w_dims[0]};
  wType.dimensions = &dims_w[0];
  NeuronModel_addOperand(model, &wType);  // Operand 1: weight
  std::shared_ptr<Node> w_node = nullptr;
  w_node = graph->Add(w_name, dims_w);
  VLOG(3) << "w_scale size: " << w_scale.size() << ",w_scale: " << w_scale[0]
          << ", wType dimensions: " << wType.dimensions[0] << " : "
          << wType.dimensions[1] << ", memory size: " << w->memory_size();

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

    biasType.dimensionCount = bias_dims.size();
    std::vector<uint32_t> dims_bias = {(uint32_t)bias_dims[0]};
    biasType.dimensions = &dims_bias[0];
    NeuronModel_addOperand(model, &biasType);  // Operand 2: bias
    bias_node = graph->Add(bias_name, dims_bias);
    VLOG(3) << "Bias name: " << bias_name << ", bias dims: " << bias_dims
            << ", bias scale: " << biasType.scale
            << " ,memory size: " << bias->memory_size();
  } else {
    biasType.dimensionCount = 1;
    std::vector<uint32_t> dims_bias = {(uint32_t)n};
    biasType.dimensions = &dims_bias[0];
    NeuronModel_addOperand(model, &biasType);  // Operand 2: bias
    bias_node = graph->Add(w_name + "_default_bias", dims_bias);
  }

  // Add fuse type
  NeuronOperandType fuseType;
  fuseType.type = NEURON_INT32;
  fuseType.dimensionCount = 0;
  std::vector<uint32_t> dims_int32 = {0};
  NeuronModel_addOperand(model, &fuseType);  // Operand 3: fuse
  std::shared_ptr<Node> fuse_node = nullptr;
  fuse_node = graph->Add(w_name + "_fuse", dims_int32);

  // Add output tensor type
  NeuronOperandType outType;
  outType.type = NEURON_TENSOR_QUANT8_ASYMM;
  outType.scale = out_scale;
  outType.zeroPoint = 128;
  outType.dimensionCount = 2;
  std::vector<uint32_t> dims_out = {(uint32_t)out_dims[0],
                                    (uint32_t)out_dims[1]};
  outType.dimensions = &dims_out[0];
  VLOG(3) << "out_scale: " << out_scale
          << ", outType: " << outType.dimensions[0] << " : "
          << outType.dimensions[1];
  NeuronModel_addOperand(model, &outType);
  std::shared_ptr<Node> out_node = nullptr;
  out_node = graph->Add(out_name, dims_out);

  int8_t* w_data = w->mutable_data<int8_t>();
  Tensor transpose_filter;
  // Original dimension
  transpose_filter.Resize({(uint32_t)w_dims[1], (uint32_t)w_dims[0]});
  transpose_filter.mutable_data<uint8_t>();
  transposeAsym(w->data<int8_t>(),
                transpose_filter.mutable_data<uint8_t>(),
                {1, 1, (uint32_t)w_dims[0], (uint32_t)w_dims[1]},
                {0, 1, 3, 2});
  memcpy(w->mutable_data<int8_t>(),
         transpose_filter.mutable_data<uint8_t>(),
         w->memory_size());
  int neuron_errCode = NeuronModel_setOperandValue(
      model, w_node->index(), w->raw_data(), w->memory_size());
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Set W operand value fail:" << neuron_errCode
                 << ",index: " << w_node->index();
    return FAILED;
  }

  // Add bias if bias tensor exists
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    int32_t* int32_bias_data =
        reinterpret_cast<int32_t*>(bias->mutable_data<float>());
    float2int32(bias->data<float>(), input_scale, w_scale, int32_bias_data);

    VLOG(3) << int32_bias_data[0] << ":" << int32_bias_data[1] << ":"
            << int32_bias_data[2] << ":" << int32_bias_data[3];
    neuron_errCode =
        NeuronModel_setOperandValue(model,
                                    bias_node->index(),
                                    bias->raw_data(),
                                    bias->memory_size());  // Operand 2: bias
  } else {
    auto int32_bias = std::make_shared<Tensor>();
    int32_bias->Resize({1, out_dims[1]});
    int32_bias->mutable_data<int32_t>();
    memset(int32_bias->mutable_data<int32_t>(), 0, int32_bias->memory_size());
    VLOG(3) << "default: " << int32_bias->memory_size();
    neuron_errCode = NeuronModel_setOperandValue(
        model,
        bias_node->index(),
        int32_bias->raw_data(),
        int32_bias->memory_size());  // Operand 2: bias
    bias_node->set_data(int32_bias);
  }
  // Add fuse value
  int32_t fuse_val[1] = {0};
  NeuronModel_setOperandValue(model,
                              fuse_node->index(),
                              fuse_val,
                              sizeof(int32_t) * 1);  // Operand 3: fuse

  std::vector<uint32_t> addInIndex = {in_node->index(),     // 0: input
                                      w_node->index(),      // 1: weight
                                      bias_node->index(),   // 2: bias
                                      fuse_node->index()};  // 3: fuse
  std::vector<uint32_t> addOutIndex = {out_node->index()};
  neuron_errCode = NeuronModel_addOperation(model,
                                            NEURON_FULLY_CONNECTED,
                                            addInIndex.size(),
                                            &addInIndex[0],
                                            addOutIndex.size(),
                                            &addOutIndex[0]);

  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Add op fail:" << op_type;
    return FAILED;
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fc, kAPU, paddle::lite::subgraph::apu::FCConverter);
