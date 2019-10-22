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

#include "lite/backends/xpu/runtime.h"
#include <vector>
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace xpu {

// Extract model data and restore model runtime
bool LoadModel(const lite::Tensor &model,
               std::shared_ptr<xtcl::network::xRuntimeInstance> *runtime) {
  LOG(INFO) << "[XPU] Load Model.";
  CHECK_GT(model.dims().production(), 0);
  std::string name(reinterpret_cast<const char *>(model.data<int8_t>()));
  LOG(INFO) << "[XPU] Model Name: " << name;
  CHECK(runtime != nullptr);
  *runtime = DeviceInfo::Global().Find(name);
  if (*runtime == nullptr) {
    LOG(WARNING) << "[XPU] Load Model failed!";
    return false;
  }
  return true;
}

}  // namespace xpu
}  // namespace lite
}  // namespace paddle
