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

#include "lite/kernels/apu/bridges/utility.h"
#include <utility>
#include "lite/kernels/apu/bridges/graph.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname) {
  auto iarg_names = op_info->input_argnames();
  if (std::find(iarg_names.begin(), iarg_names.end(), argname) !=
      iarg_names.end()) {
    auto inputs = op_info->Input(argname);
    if (inputs.empty()) {
      return false;
    }
    auto var_name = inputs.front();
    auto var = scope->FindVar(var_name);
    return var != nullptr;
  } else {
    return false;
  }
}

void insert_transpose_node(void* ctx,
                           const std::string& input_name,
                           const std::string& output_name,
                           std::vector<uint32_t> input_shape,
                           std::vector<uint32_t> output_shape,
                           std::vector<int32_t> axis,
                           float scale,
                           int32_t zeroPoint) {
  int neuron_errCode;
  auto graph = static_cast<Graph*>(ctx);
  auto model = graph->model();

  // Add input
  NeuronOperandType inType;
  inType.type = NEURON_TENSOR_QUANT8_ASYMM;
  inType.scale = scale;
  inType.zeroPoint = zeroPoint;
  inType.dimensionCount = input_shape.size();
  inType.dimensions = &input_shape[0];

  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    VLOG(3) << "Has " << input_name;
    input_node = graph->Get(input_name);
  } else {
    neuron_errCode = NeuronModel_addOperand(model, &inType);  // input
    if (NEURON_NO_ERROR != neuron_errCode) {
      LOG(WARNING) << "Insert transpose op fail!";
      return;
    }
    VLOG(3) << "Add " << input_name;
    input_node = graph->Add(input_name, input_shape);
  }

  // Add perm
  NeuronOperandType permsType;
  permsType.type = NEURON_TENSOR_INT32;
  permsType.dimensionCount = 1;
  uint32_t dims_perms[1] = {4};
  permsType.dimensions = dims_perms;

  neuron_errCode = NeuronModel_addOperand(model, &permsType);  // perm
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Insert transpose op fail!";
    return;
  }
  std::shared_ptr<Node> perms_node = nullptr;
  perms_node = graph->Add(input_name + "_perms", {4});

  VLOG(3) << "axis :" << axis[0] << ":" << axis[1] << ":" << axis[2] << ":"
          << axis[3];
  //  &axis[0], sizeof(int32_t) * axis.size());
  neuron_errCode = NeuronModel_setOperandValue(
      model, perms_node->index(), &axis[0], sizeof(int32_t) * axis.size());
  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Insert transpose op fail!";
    return;
  }

  // Add output
  NeuronOperandType outType;
  outType.type = NEURON_TENSOR_QUANT8_ASYMM;
  outType.scale = scale;
  outType.zeroPoint = zeroPoint;
  outType.dimensionCount = output_shape.size();
  outType.dimensions = &output_shape[0];

  NeuronModel_addOperand(model, &outType);  // output
  std::shared_ptr<Node> output_node = nullptr;
  output_node = graph->Add(output_name, output_shape);

  std::vector<uint32_t> addInIndex = {input_node->index(),   // 0: input
                                      perms_node->index()};  // 1: perm

  std::vector<uint32_t> addOutIndex = {output_node->index()};

  neuron_errCode = NeuronModel_addOperation(model,
                                            NEURON_TRANSPOSE,
                                            addInIndex.size(),
                                            &addInIndex[0],
                                            addOutIndex.size(),
                                            &addOutIndex[0]);

  if (NEURON_NO_ERROR != neuron_errCode) {
    LOG(WARNING) << "Insert transpose op fail!";
  }
}

void transpose(const int8_t* input_data,
               uint8_t* output_data,
               std::vector<uint32_t> input_shape,
               std::vector<uint32_t> axis) {
  int old_index = -1;
  int new_index = -1;
  int dim[4] = {0};
  std::vector<uint32_t> shape = input_shape;
  VLOG(3) << input_shape[0] << ":" << input_shape[1] << ":" << input_shape[2]
          << ":" << input_shape[3];
  VLOG(3) << axis[0] << ":" << axis[1] << ":" << axis[2] << ":" << axis[3];
  for (dim[0] = 0; dim[0] < input_shape[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < input_shape[1]; dim[1]++) {
      for (dim[2] = 0; dim[2] < input_shape[2]; dim[2]++) {
        for (dim[3] = 0; dim[3] < input_shape[3]; dim[3]++) {
          old_index = dim[0] * shape[1] * shape[2] * shape[3] +
                      dim[1] * shape[2] * shape[3] + dim[2] * shape[3] + dim[3];
          new_index =
              dim[axis[0]] * shape[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[2]] * shape[axis[3]] + dim[axis[3]];

          output_data[new_index] = input_data[old_index];
        }
      }
    }
  }
}

void transposeAsym(const int8_t* input_data,
                   uint8_t* output_data,
                   std::vector<uint32_t> input_shape,
                   std::vector<uint32_t> axis) {
  int old_index = -1;
  int new_index = -1;
  int dim[4] = {0};
  std::vector<uint32_t> shape = input_shape;
  VLOG(3) << input_shape[0] << ":" << input_shape[1] << ":" << input_shape[2]
          << ":" << input_shape[3];
  VLOG(3) << axis[0] << ":" << axis[1] << ":" << axis[2] << ":" << axis[3];
  for (dim[0] = 0; dim[0] < input_shape[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < input_shape[1]; dim[1]++) {
      for (dim[2] = 0; dim[2] < input_shape[2]; dim[2]++) {
        for (dim[3] = 0; dim[3] < input_shape[3]; dim[3]++) {
          old_index = dim[0] * shape[1] * shape[2] * shape[3] +
                      dim[1] * shape[2] * shape[3] + dim[2] * shape[3] + dim[3];
          new_index =
              dim[axis[0]] * shape[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[2]] * shape[axis[3]] + dim[axis[3]];

          output_data[new_index] = input_data[old_index] + 128;  // per layer
        }
      }
    }
  }
}

void float2int32(const float* bias_data,
                 float input_scale,
                 std::vector<float> weight_scale,
                 int32_t* int32_bias_data) {
  for (int i = 0; i < weight_scale.size(); i++) {
    int32_bias_data[i] = bias_data[i] / (input_scale * weight_scale[i]);
  }
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
