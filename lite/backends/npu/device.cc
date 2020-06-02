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

bool WriteToOMFile(const domi::ModelBufferData& om_model_buff,
                   std::string om_file_path) {
  FILE* fp;
  fp = fopen(om_file_path.c_str(), "wb");
  CHECK(fp != nullptr) << om_file_path << " open failed!";

  uint32_t write_size =
      (uint32_t)fwrite(om_model_buff.data, 1, om_model_buff.length, fp);
  CHECK_EQ(write_size, om_model_buff.length) << "write om file failed !";

  fclose(fp);
  return true;
}

bool ReadFromOMFile(domi::ModelBufferData* om_model_buff,
                    std::string om_file_path) {
  FILE* fp;
  fp = fopen(om_file_path.c_str(), "rb");
  CHECK(fp != nullptr) << om_file_path << " open failed!";

  fseek(fp, 0, SEEK_END);
  uint32_t model_length = (uint32_t)ftell(fp);
  fseek(fp, 0, SEEK_SET);
  om_model_buff->data = malloc(model_length);
  om_model_buff->length = model_length;
  uint32_t read_size =
      (uint32_t)fread(om_model_buff->data, 1, model_length, fp);
  CHECK_EQ(read_size, model_length) << "read om file failed !";

  fclose(fp);
  return true;
}

std::shared_ptr<hiai::AiModelMngerClient> Device::Build(
    const std::string model_name,                // NOLINT
    std::vector<ge::Operator>& input_nodes,      // NOLINT
    std::vector<ge::Operator>& output_nodes,     // NOLINT
    const std::string model_cache_full_dir = ""  // NOLINT
    ) {
  VLOG(3) << "[NPU] Build model";
  // Build the HiAI IR graph to the HiAI om model
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  ge::Model om_model("model", "model");
  om_model.SetGraph(ir_graph);
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData om_model_buf;

  if (!model_cache_full_dir.empty() && IsFileExists(model_cache_full_dir)) {
    VLOG(3) << "Will read om model from " << model_cache_full_dir;
    ReadFromOMFile(&om_model_buf, model_cache_full_dir);
  } else {
    if (!ir_build.CreateModelBuff(om_model, om_model_buf)) {
      LOG(WARNING) << "[NPU] CreateModelBuff failed!";
      return nullptr;
    }
    if (!ir_build.BuildIRModel(om_model, om_model_buf)) {
      LOG(WARNING) << "[NPU] BuildIRModel failed!";
      ir_build.ReleaseModelBuff(om_model_buf);
      return nullptr;
    }
    if (!model_cache_full_dir.empty()) {
      VLOG(3) << "Will write om model to " << model_cache_full_dir;
      WriteToOMFile(om_model_buf, model_cache_full_dir);
    }
  }

  // Create a HiAI model manager client to load the HiAI om model
  std::shared_ptr<hiai::AiModelMngerClient> model_client(
      new hiai::AiModelMngerClient());
  if (model_client->Init(nullptr) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient init failed)!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return nullptr;
  }
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
