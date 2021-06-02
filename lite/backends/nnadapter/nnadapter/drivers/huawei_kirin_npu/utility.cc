// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "utility.h"                  // NOLINT
#include "../../nnadapter_logging.h"  // NOLINT

namespace nnadapter {
namespace driver {
namespace huawei_kirin_npu {

std::shared_ptr<hiai::AiModelMngerClient> LoadOMModelFromBuffer(
    const std::string& model_name,
    std::vector<char>* model_buffer,
    bool* model_comp,
    int freq_level,
    int framework_type,
    int model_type,
    int device_type) {
  // Create a hiai model manager client to load a hiai om model
  auto model_client = std::make_shared<hiai::AiModelMngerClient>();
  if (model_client->Init(nullptr) != hiai::AI_SUCCESS) {
    NNADAPTER_LOG(WARNING) << "Init hiai model client failed!";
    return nullptr;
  }
  // Check hiai DDK version
  const char* ddk_version = model_client->GetVersion();
  if (ddk_version) {
    NNADAPTER_VLOG(3) << "hiai DDK version: " << ddk_version;
  } else {
    NNADAPTER_LOG(WARNING) << "Unable to get hiai DDK version!";
  }
  // Check model compatibility
  auto model_desc = std::make_shared<hiai::AiModelDescription>(
      model_name, freq_level, framework_type, model_type, device_type);
  model_desc->SetModelBuffer(
      reinterpret_cast<const void*>(model_buffer->data()),
      model_buffer->size());
  if (!*model_comp &&
      model_client->CheckModelCompatibility(*model_desc, *model_comp) !=
          hiai::AI_SUCCESS) {
    *model_comp = false;
    NNADAPTER_VLOG(3)
        << "hiai om model is NOT compatiblitiable, set model_comp to "
        << *model_comp;
  } else {
    *model_comp = true;
    NNADAPTER_VLOG(3) << "hiai om model is compatiblitiable, set model_comp to "
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
          framework_type, org_model_buffers);
      // NNADAPTER_VLOG(3) << "new hiai om model buffer memeory size is " <<
      // new_model_buffer->GetMemBufferSize();
      if (new_model_buffer) {
        uint32_t new_model_size = 0;
        if (model_builder->BuildModel(org_model_buffers,
                                      new_model_buffer,
                                      new_model_size) == hiai::AI_SUCCESS) {
          // Need to change to new_model_size as GetMemBufferSize is not
          // correct.
          model_buffer->resize(new_model_size);
          memcpy(reinterpret_cast<void*>(model_buffer->data()),
                 new_model_buffer->GetMemBufferData(),
                 new_model_size);
          // Reset the model buffer
          model_desc->SetModelBuffer(
              reinterpret_cast<const void*>(model_buffer->data()),
              model_buffer->size());
          NNADAPTER_VLOG(3) << "Rebuild the compatible model done.";
        } else {
          NNADAPTER_LOG(WARNING)
              << "Failed to call BuildModel to rebuild the compatible model!";
        }
        model_builder->MemBufferDestroy(new_model_buffer);
      } else {
        NNADAPTER_LOG(WARNING) << "Failed to call OutputMemBufferCreate for "
                                  "storing a new compatiable hiai om model!";
      }
      model_builder->MemBufferDestroy(org_model_buffer);
    } else {
      NNADAPTER_LOG(WARNING) << "Failed to call InputMemBufferCreate for "
                                "writing an old compatiable hiai om model!";
    }
  }
  // Load the compatible model
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs{
      model_desc};
  if (model_client->Load(model_descs) != hiai::AI_SUCCESS) {
    NNADAPTER_LOG(WARNING)
        << "Failed to call AiModelMngerClient to load hiai om model!";
    return nullptr;
  }
  NNADAPTER_VLOG(3) << "Load successed.";
  return model_client;
}

bool BuildOMModelToBuffer(std::vector<ge::Operator>& input_nodes,   // NOLINT
                          std::vector<ge::Operator>& output_nodes,  // NOLINT
                          std::vector<char>* model_buffer) {
  // Convert a hiai IR graph to a hiai om model
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  ge::Model om_model("model", "model");
  om_model.SetGraph(ir_graph);

  // Build a hiai om model and serialize it into a om buffer
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData om_buffer;
  if (!ir_build.CreateModelBuff(om_model, om_buffer)) {
    NNADAPTER_LOG(WARNING)
        << "Failed to call CreateModelBuff for storing the om model!";
    return false;
  }
  if (!ir_build.BuildIRModel(om_model, om_buffer)) {
    NNADAPTER_LOG(WARNING) << "Failed to call BuildIRModel for converting a "
                              "hiai IR graph to a hiai om model!";
    ir_build.ReleaseModelBuff(om_buffer);
    return false;
  }
  model_buffer->resize(om_buffer.length);
  memcpy(reinterpret_cast<void*>(model_buffer->data()),
         reinterpret_cast<void*>(om_buffer.data),
         om_buffer.length);
  ir_build.ReleaseModelBuff(om_buffer);
  NNADAPTER_VLOG(3) << "Build succeeded.";
  return true;
}

}  // namespace huawei_kirin_npu
}  // namespace driver
}  // namespace nnadapter
