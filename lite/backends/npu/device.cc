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

bool WriteToOMFile(const domi::ModelBufferData& om_model_buf,
                   std::string om_file_path) {
  FILE* fp = fopen(om_file_path.c_str(), "wb");
  if (!fp) {
    LOG(WARNING) << "[NPU] Open om model file " << om_file_path
                 << " for writting failed!";
    return false;
  }
  uint32_t written_size =
      (uint32_t)fwrite(om_model_buf.data, 1, om_model_buf.length, fp);
  CHECK_EQ(written_size, om_model_buf.length) << "[NPU] Write om file "
                                              << om_file_path << " failed!";
  fclose(fp);
  return true;
}

bool ReadFromOMFile(domi::ModelBufferData* om_model_buf,
                    std::string om_file_path) {
  FILE* fp = fopen(om_file_path.c_str(), "rb");
  if (!fp) {
    LOG(WARNING) << "[NPU] Open om model file " << om_file_path
                 << " for reading failed!";
    return false;
  }
  fseek(fp, 0, SEEK_END);
  uint32_t model_length = (uint32_t)ftell(fp);
  fseek(fp, 0, SEEK_SET);
  om_model_buf->data = malloc(model_length);
  CHECK(om_model_buf->data != nullptr);
  om_model_buf->length = model_length;
  uint32_t read_size = (uint32_t)fread(om_model_buf->data, 1, model_length, fp);
  CHECK_EQ(read_size, model_length) << "[NPU] Read om model file "
                                    << om_file_path << " failed!";
  fclose(fp);
  return true;
}

std::shared_ptr<hiai::AiModelMngerClient> Device::Build(
    const std::string model_name, domi::ModelBufferData* model_buffer) {
  // Create a HiAI model manager client to load the HiAI om model
  std::shared_ptr<hiai::AiModelMngerClient> model_client(
      new hiai::AiModelMngerClient());
  if (model_client->Init(nullptr) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient init failed!";
    return nullptr;
  }
  auto model_desc = std::make_shared<hiai::AiModelDescription>(
      model_name, freq_level(), framework_type(), model_type(), device_type());
  model_desc->SetModelBuffer(model_buffer->data, model_buffer->length);
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs;
  model_descs.push_back(model_desc);
  if (model_client->Load(model_descs) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient load model failed!";
    return nullptr;
  }
  VLOG(3) << "[NPU] Build done.";
  return model_client;
}

std::shared_ptr<hiai::AiModelMngerClient> Device::Build(
    const std::string& model_name, const std::string& model_cache_dir) {
  auto model_path = model_cache_dir + "/" + model_name;
  VLOG(3) << "[NPU] Build from om model file " << model_path;
  domi::ModelBufferData model_buffer;
  if (!ReadFromOMFile(&model_buffer, model_path)) {
    return nullptr;
  }
  auto model_client = Build(model_name, &model_buffer);
  domi::HiaiIrBuild ir_build;
  ir_build.ReleaseModelBuff(model_buffer);
  return model_client;
}

std::shared_ptr<hiai::AiModelMngerClient> Device::Build(
    const std::string& model_name,
    std::vector<ge::Operator>& input_nodes,   // NOLINT
    std::vector<ge::Operator>& output_nodes,  // NOLINT
    const std::string& model_cache_dir        // NOLINT
    ) {
  VLOG(3) << "[NPU] Build from IR graph inputs(" << input_nodes.size()
          << ") outputs(" << output_nodes.size() << ")";
  // Convert the HiAI IR graph to the HiAI om model
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  ge::Model om_model("model", "model");
  om_model.SetGraph(ir_graph);

  // Build the HiAI om model, serialize and output it to the om buffer
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData model_buffer;
  if (!ir_build.CreateModelBuff(om_model, model_buffer)) {
    LOG(WARNING) << "[NPU] CreateModelBuff failed!";
    return nullptr;
  }
  if (!ir_build.BuildIRModel(om_model, model_buffer)) {
    LOG(WARNING) << "[NPU] BuildIRModel failed!";
    ir_build.ReleaseModelBuff(model_buffer);
    return nullptr;
  }

  // Invoke the HiAI service to load om model buffer and return the model client
  // object
  auto model_client = Build(model_name, &model_buffer);
  if (model_client != nullptr && !model_cache_dir.empty()) {
    auto model_path = model_cache_dir + "/" + model_name;
    VLOG(3) << "[NPU] Save to om model file " << model_path;
    WriteToOMFile(model_buffer, model_path);
  }
  ir_build.ReleaseModelBuff(model_buffer);
  return model_client;
}

}  // namespace npu
}  // namespace lite
}  // namespace paddle
