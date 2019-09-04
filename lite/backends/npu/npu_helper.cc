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

#include "lite/backends/npu/npu_helper.h"
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ai_ddk_lib/include/HiAiModelManagerService.h"
#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/hiai_ir_build.h"

namespace paddle {
namespace lite {
namespace npu {

bool SaveNPUModel(const void* om_model_data,
                  const size_t om_model_size,
                  const std::string& om_file_path) {
  std::FILE* fp;
  fp = std::fopen(om_file_path.c_str(), "wb");
  if (fp == NULL) {
    LOG(WARNING) << "[NPU] " << om_file_path << " open failed!";
    return false;
  }

  size_t write_size = std::fwrite(om_model_data, 1, om_model_size, fp);
  if (write_size != om_model_size) {
    std::fclose(fp);
    LOG(WARNING) << "[NPU] Write NPU model failed: " << om_file_path;
    return false;
  }
  std::fclose(fp);
  return true;
}

bool BuildNPUClient(const void* om_model_data,
                    const size_t om_model_size,
                    const std::string& name) {
  std::unique_ptr<hiai::AiModelMngerClient> client(
      new hiai::AiModelMngerClient);
  int ret = client->Init(nullptr);
  if (ret != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] Failed building NPU client " << name
                 << ", ret: " << ret;
    throw std::runtime_error("");
    return false;
  }

  auto desc = std::make_shared<hiai::AiModelDescription>(
      name,
      DeviceInfo::Global().freq_level(),
      DeviceInfo::Global().framework_type(),
      DeviceInfo::Global().model_type(),
      DeviceInfo::Global().device_type());
  desc->SetModelBuffer(om_model_data, om_model_size);

  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_desc;
  model_desc.push_back(desc);
  if (client->Load(model_desc) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] Model Load Failed: " << desc->GetName();
    throw std::runtime_error("");
    return false;
  }

  DeviceInfo::Global().Insert(name, std::move(client));
  return true;
}

// If build from inputs and outputs will save the npu offline model
bool BuildNPUClient(std::vector<ge::Operator>& inputs,   // NOLINT
                    std::vector<ge::Operator>& outputs,  // NOLINT
                    const std::string& name) {
  LOG(INFO) << "[NPU] Building Client";
  ge::Graph npu_subgraph("npu_subgraph" + name);
  npu_subgraph.SetInputs(inputs).SetOutputs(outputs);

  ge::Model npu_model("model", "npu_model" + name);
  npu_model.SetGraph(npu_subgraph);

  // compile IR graph and output om model to memory
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData om_model_buffer;
  if (!ir_build.CreateModelBuff(npu_model, om_model_buffer)) {
    LOG(WARNING) << "[NPU] Failed CreateModelBuff: " << npu_model.GetName();
    return false;
  }
  if (!ir_build.BuildIRModel(npu_model, om_model_buffer)) {
    LOG(WARNING) << "[NPU] Failed BuildIRModel: " << npu_model.GetName();
    return false;
  }

  if (BuildNPUClient(om_model_buffer.data, om_model_buffer.length, name)) {
    // save npu offline model
    if (!SaveNPUModel(om_model_buffer.data, om_model_buffer.length, name)) {
      LOG(WARNING) << "[NPU] Save model " << name << " failed.";
    }
    ir_build.ReleaseModelBuff(om_model_buffer);
    return true;
  }
  return false;
}

// If build from path will not save the npu offline model
bool BuildNPUClient(const std::string& om_model_file_path,
                    const std::string& name) {
  // load om model from file
  std::ifstream file(om_model_file_path, std::ios::binary);
  CHECK(file.is_open()) << "[NPU] Unable to open om model file: "
                        << om_model_file_path;
  const auto fbegin = file.tellg();
  file.seekg(0, std::ios::end);
  const auto fend = file.tellg();
  size_t om_model_size = fend - fbegin;
  VLOG(5) << "[NPU] om model file size: " << om_model_size;
  file.seekg(0, std::ios::beg);
  std::vector<char> om_model_data(om_model_size);
  file.read(om_model_data.data(), om_model_size);

  return BuildNPUClient(
      reinterpret_cast<void*>(om_model_data.data()), om_model_size, name);
}

}  // namespace npu
}  // namespace lite
}  // namespace paddle
