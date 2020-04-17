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

#include <dlfcn.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "NeuronAdapter.h"
#include "lite/core/op_lite.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

// typedef to the build functions pointer signatures
typedef int (*Neuron_getVersion)(uint32_t* version);
typedef int (*NeuronModel_create)(NeuronModel** model);
typedef void (*NeuronModel_free)(NeuronModel* model);
typedef int (*NeuronModel_finish)(NeuronModel* model);
typedef int (*NeuronModel_addOperand)(NeuronModel* model,
                                      const NeuronOperandType* type);
typedef int (*NeuronModel_setOperandValue)(NeuronModel* model,
                                           int32_t index,
                                           const void* buffer,
                                           size_t length);
typedef int (*NeuronModel_addOperation)(NeuronModel* model,
                                        NeuronOperationType type,
                                        uint32_t inputCount,
                                        const uint32_t* inputs,
                                        uint32_t outputCount,
                                        const uint32_t* outputs);
typedef int (*NeuronModel_identifyInputsAndOutputs)(NeuronModel* model,
                                                    uint32_t inputCount,
                                                    const uint32_t* inputs,
                                                    uint32_t outputCount,
                                                    const uint32_t* outputs);
typedef int (*NeuronModel_setOperandSymmPerChannelQuantParams)(
    NeuronModel* model,
    int32_t index,
    const NeuronSymmPerChannelQuantParams* channelQuant);
typedef int (*NeuronExecution_create)(NeuronCompilation* compilation,
                                      NeuronExecution** execution);
typedef void (*NeuronExecution_free)(NeuronExecution* execution);
typedef int (*NeuronExecution_setInput)(NeuronExecution* execution,
                                        int32_t index,
                                        const NeuronOperandType* type,
                                        const void* buffer,
                                        size_t length);
typedef int (*NeuronExecution_setOutput)(NeuronExecution* execution,
                                         int32_t index,
                                         const NeuronOperandType* type,
                                         void* buffer,
                                         size_t length);
typedef int (*NeuronExecution_compute)(NeuronExecution* execution);

void* LoadFunc(void* libHandle, const char* name);

#define LOAD_FUNCTIONS(libHandle, FUNC_NAME, VARIABLE_NAME) \
  FUNC_NAME VARIABLE_NAME =                                 \
      reinterpret_cast<FUNC_NAME>(LoadFunc(libHandle, #FUNC_NAME));

// Type/tensor converters for converting Paddle type/tensor to HiAI type/tensor
bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

void insert_transpose_node(void* ctx,
                           const std::string& input_name,
                           const std::string& output_name,
                           std::vector<uint32_t> input_shape,
                           std::vector<uint32_t> output_shape,
                           std::vector<int32_t> axis,
                           float scale,
                           int32_t zeroPoint);

void transpose(const int8_t* input_data,
               uint8_t* output_data,
               std::vector<uint32_t> input_shape,
               std::vector<uint32_t> axis);

void transposeAsym(const int8_t* input_data,
                   uint8_t* output_data,
                   std::vector<uint32_t> input_shape,
                   std::vector<uint32_t> axis);

void float2int32(const float* bias_data,
                 float input_scale,
                 std::vector<float> weight_scale,
                 int32_t* int32_bias_data);

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
