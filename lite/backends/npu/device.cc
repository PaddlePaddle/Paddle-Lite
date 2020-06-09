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

#include "lite/backends/npu/device.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/io.h"

namespace paddle {
namespace lite {
namespace npu {

std::shared_ptr<hiai::AiModelMngerClient> Device::Load(
    const std::string& model_name, const std::vector<char>& model_buffer) {
  // Create a HiAI model manager client to load the HiAI om model
  auto model_client = std::make_shared<hiai::AiModelMngerClient>();
  if (model_client->Init(nullptr) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient init failed!";
    return nullptr;
  }
  auto model_desc = std::make_shared<hiai::AiModelDescription>(
      model_name, freq_level(), framework_type(), model_type(), device_type());
  model_desc->SetModelBuffer(reinterpret_cast<const void*>(model_buffer.data()),
                             model_buffer.size());
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs{
      model_desc};
  if (model_client->Load(model_descs) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient load model failed!";
    return nullptr;
  }
  VLOG(3) << "[NPU] Load model done.";
  return model_client;
}

bool Device::Build(std::vector<ge::Operator>& input_nodes,   // NOLINT
                   std::vector<ge::Operator>& output_nodes,  // NOLINT
                   std::vector<char>* model_buffer) {
  // Convert the HiAI IR graph to the HiAI om model
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  ge::Model om_model("model", "model");
  om_model.SetGraph(ir_graph);

  // Build the HiAI om model, serialize and output it to the om buffer
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData om_buffer;
  if (!ir_build.CreateModelBuff(om_model, om_buffer)) {
    LOG(WARNING) << "[NPU] CreateModelBuff failed!";
    return false;
  }
  if (!ir_build.BuildIRModel(om_model, om_buffer)) {
    LOG(WARNING) << "[NPU] BuildIRModel failed!";
    ir_build.ReleaseModelBuff(om_buffer);
    return false;
  }
  model_buffer->resize(om_buffer.length);
  memcpy(reinterpret_cast<void*>(model_buffer->data()),
         reinterpret_cast<void*>(om_buffer.data),
         om_buffer.length);
  ir_build.ReleaseModelBuff(om_buffer);
  VLOG(3) << "[NPU] Build model done.";
  return true;
}

}  // namespace npu
}  // namespace lite
}  // namespace paddle
