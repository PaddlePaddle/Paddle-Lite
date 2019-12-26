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

namespace paddle {
namespace lite {
namespace npu {

std::unique_ptr<hiai::AiModelMngerClient> Device::Build(
    std::string& model_name,                 // NOLINT
    std::vector<ge::Operator>& input_nodes,  // NOLINT
    std::vector<ge::Operator>& output_nodes  // NOLINT
    ) {
  VLOG(3) << "[NPU] Build model";
  // Build the HiAI IR graph to the HiAI om model
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  ge::Model om_model("model", "model");
  om_model.SetGraph(ir_graph);
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData om_model_buf;
  if (!ir_build.CreateModelBuff(om_model, om_model_buf)) {
    LOG(WARNING) << "[NPU] CreateModelBuff failed!";
    return nullptr;
  }
  if (!ir_build.BuildIRModel(om_model, om_model_buf)) {
    LOG(WARNING) << "[NPU] BuildIRModel failed!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return nullptr;
  }
  // Create a HiAI model manager client to load the HiAI om model
  std::unique_ptr<hiai::AiModelMngerClient> model_client(
      new hiai::AiModelMngerClient());
  if (model_client->Init(nullptr) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient init failed)!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return nullptr;
  }
  model_name = "model_" + std::to_string(model_count_++) + ".om";
  auto model_desc = std::make_shared<hiai::AiModelDescription>(
      model_name, freq_level(), framework_type(), model_type(), device_type());
  model_desc->SetModelBuffer(om_model_buf.data, om_model_buf.length);
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs;
  model_descs.push_back(model_desc);
  if (model_client->Load(model_descs) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient load model failed!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return nullptr;
  }
  ir_build.ReleaseModelBuff(om_model_buf);
  VLOG(3) << "[NPU] Build done";
  return model_client;
}

}  // namespace npu
}  // namespace lite
}  // namespace paddle
