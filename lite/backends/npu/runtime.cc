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

#include "lite/backends/npu/runtime.h"
#include <string>
#include <vector>
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace npu {

// Create hiai model manager to load om model from lite tensor, and return the
// manager and an unique model name
bool LoadModel(const lite::Tensor &model_data,
               std::shared_ptr<hiai::AiModelMngerClient> *model_client,
               std::string *model_name) {
  LOG(INFO) << "[NPU] Load model.";
  auto model_data_ptr = model_data.data<int8_t>();
  auto model_data_size = model_data.numel() * sizeof(int8_t);
  if (model_data_ptr == nullptr || model_data_size == 0) {
    return false;
  }
  *model_client = std::make_shared<hiai::AiModelMngerClient>();
  int ret = (*model_client)->Init(nullptr);
  if (ret != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient init failed(" << ret << ")!";
    return false;
  }
  *model_name = "model.om";
  auto model_desc = std::make_shared<hiai::AiModelDescription>(
      *model_name,
      DeviceInfo::Global().freq_level(),
      DeviceInfo::Global().framework_type(),
      DeviceInfo::Global().model_type(),
      DeviceInfo::Global().device_type());
  model_desc->SetModelBuffer(model_data_ptr, model_data_size);
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs;
  model_descs.push_back(model_desc);
  if ((*model_client)->Load(model_descs) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient load model failed!";
    return false;
  }
  return true;
}

}  // namespace npu
}  // namespace lite
}  // namespace paddle
