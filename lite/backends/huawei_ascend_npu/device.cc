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

#include "lite/backends/huawei_ascend_npu/device.h"
#include <map>
#include <utility>
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"
#include "graph/graph.h"
#include "lite/utils/io.h"

namespace paddle {
namespace lite {
namespace huawei_ascend_npu {

std::shared_ptr<AclModelClient> Device::LoadFromMem(
    const std::vector<char>& model_buffer) {
  if (model_buffer.size() == 0) {
    LOG(ERROR) << "[HUAWEI_ASCEND_NPU] model_buffer size is ZERO!";
    return nullptr;
  }

  // Create a ACL model  client to load the om model
  std::shared_ptr<AclModelClient> model_client(new AclModelClient());
  // Load model from memory
  if (model_client->LoadFromMem(
          reinterpret_cast<const void*>(model_buffer.data()),
          model_buffer.size())) {
    return model_client;
  }
  return nullptr;
}

std::shared_ptr<AclModelClient> Device::LoadFromFile(
    const std::string& model_path) {
  if (!paddle::lite::IsFileExists(model_path)) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] om model file not exists:" << model_path;
    return nullptr;
  }

  // Create a ACL model  client to load the om model
  std::shared_ptr<AclModelClient> model_client(new AclModelClient());
  // Load model from memory
  if (model_client->LoadFromFile(model_path.c_str())) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] Loading model file success:" << model_path;
    return model_client;
  }
  return nullptr;
}

bool Device::Build(std::vector<ge::Operator>& input_nodes,   // NOLINT
                   std::vector<ge::Operator>& output_nodes,  // NOLINT
                   std::vector<char>* model_buffer) {
  // Convert the HiAI IR graph to the HiAI om model
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);

  // Build IR model
  ge::ModelBufferData om_buffer;
  std::map<std::string, std::string> options;
  options.insert(std::make_pair(ge::ir_option::LOG_LEVEL, "error"));

  ATC_CALL(aclgrphBuildModel(ir_graph, options, om_buffer));

  // Copy from om model buffer
  model_buffer->resize(om_buffer.length);
  memcpy(reinterpret_cast<void*>(model_buffer->data()),
         reinterpret_cast<void*>(om_buffer.data.get()),
         om_buffer.length);

  return true;
}

}  // namespace huawei_ascend_npu
}  // namespace lite
}  // namespace paddle
