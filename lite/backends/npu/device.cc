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
    const std::string& model_name,
    std::vector<char>* model_buffer,
    bool* model_comp) {
  // Create a HiAI model manager client to load the HiAI om model
  auto model_client = std::make_shared<hiai::AiModelMngerClient>();
  if (model_client->Init(nullptr) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] Init hiai model client failed!";
    return nullptr;
  }
  // Check HiAI DDK version
  const char* ddk_version = model_client->GetVersion();
  if (ddk_version) {
    VLOG(3) << "[NPU] HiAI DDK version: " << ddk_version;
  } else {
    LOG(WARNING) << "[NPU] Unable to get HiAI DDK version!";
  }
  // Check model compatibility
  auto model_desc = std::make_shared<hiai::AiModelDescription>(
      model_name, freq_level(), framework_type(), model_type(), device_type());
  model_desc->SetModelBuffer(
      reinterpret_cast<const void*>(model_buffer->data()),
      model_buffer->size());
  if (!*model_comp &&
      model_client->CheckModelCompatibility(*model_desc, *model_comp) !=
          hiai::AI_SUCCESS) {
    *model_comp = false;
    VLOG(3) << "[NPU] model is NOT compatiblitiable, setting model_comp to "
            << *model_comp;
  } else {
    *model_comp = true;
    VLOG(3) << "[NPU] model is compatiblitiable, setting model_comp to "
            << *model_comp;
  }
  // Rebuild and write the data of the compatible model to the model buffer
  if (!*model_comp) {
    std::shared_ptr<hiai::AiModelBuilder> model_builder =
        std::make_shared<hiai::AiModelBuilder>(model_client);
    hiai::MemBuffer* org_model_buffer = model_builder->InputMemBufferCreate(
        reinterpret_cast<void*>(model_buffer->data()), model_buffer->size());
    if (org_model_buffer) {
      std::vector<hiai::MemBuffer*> org_model_buffers;
      org_model_buffers.push_back(org_model_buffer);
      hiai::MemBuffer* new_model_buffer = model_builder->OutputMemBufferCreate(
          framework_type(), org_model_buffers);
      // VLOG(3) << "[NPU] new model buffer memeory size is " <<
      // new_model_buffer->GetMemBufferSize();
      if (new_model_buffer) {
        uint32_t new_model_size = 0;
        if (model_builder->BuildModel(org_model_buffers,
                                      new_model_buffer,
                                      new_model_size) == hiai::AI_SUCCESS) {
          // need to change to new_model_size as GetMemBufferSize is not
          // correct.
          model_buffer->resize(new_model_size);
          memcpy(reinterpret_cast<void*>(model_buffer->data()),
                 new_model_buffer->GetMemBufferData(),
                 new_model_size);
          // Reset the model buffer
          model_desc->SetModelBuffer(
              reinterpret_cast<const void*>(model_buffer->data()),
              model_buffer->size());
          VLOG(3) << "[NPU] Rebuild the compatible model done.";
        } else {
          LOG(WARNING) << "[NPU] Rebuild the compatible model failed!";
        }
        model_builder->MemBufferDestroy(new_model_buffer);
      } else {
        LOG(WARNING) << "[NPU] OutputMemBufferCreate failed!";
      }
      model_builder->MemBufferDestroy(org_model_buffer);
    } else {
      LOG(WARNING) << "[NPU] InputMemBufferCreate failed!";
    }
  }
  // Load the compatible model
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
