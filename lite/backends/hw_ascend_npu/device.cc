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

#include "lite/backends/hw_ascend_npu/device.h"
#include <map>
#include <string>
#include "ge/ge_api_types.h"
#include "lite/backends/hw_ascend_npu/build.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace hw_ascend_npu {
std::shared_ptr<HWAscendNPURuntime> Device::Build(
    std::vector<ge::Operator>& input_nodes,  // NOLINT
    std::vector<ge::Operator>& output_nodes  // NOLINT
    ) {
  std::shared_ptr<ge::ModelBufferData> model_data =
      paddle::lite::hw_ascend_npu::Build(input_nodes, output_nodes);
  if (model_data == nullptr) {
    LOG(ERROR) << "[HWAscendNPU] Build model failed";
    return nullptr;
  }
  LOG(INFO) << "[HWAscendNPU] Build model success";

  if (!inited_) {
    if (0 == InitDevice()) {
      LOG(INFO) << "Init success.";
      inited_ = true;
    }
  }
  std::shared_ptr<HWAscendNPURuntime> model_runtime(
      new HWAscendNPURuntime(model_data->data, model_data->length));
  CHECK(model_runtime != nullptr);
  if (!model_runtime->model_loaded()) {
    LOG(ERROR) << "[HWAscendNPU]: Can not create model runtime instance";
    return nullptr;
  }
  LOG(INFO) << "[HWAscendNPU]: Build done";
  return model_runtime;
}

int Device::InitDevice() {
  const char* acl_conf = "/usr/local/acl.json";
  aclError ret = aclInit(acl_conf);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU] acl init failed";
    return -1;
  }

  // open device
  ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU] acl open device " << device_id_ << " failed";
    return -1;
  }

  ret = aclrtCreateContext(&context_ptr_, device_id_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "acl create context failed";
    return -1;
  }

  // create stream
  ret = aclrtCreateStream(&stream_ptr_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU] acl create stream failed";
    return -1;
  }

  // get run mode
  aclrtGetRunMode runMode;
  ret = aclrtGetMode(&runMode);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "[HWAscendNPU] acl get run mode failed";
    return -1;
  }
  is_devcie_ = (runMode == ACL_DEVICE);
  LOG(INFO) << "[HWAscendNPU] Hardware initialization done";
  return 0;
}

void Device::ReleaseDevice() {
  aclError ret;
  if (stream_ptr_ != nullptr) {
    ret = aclrtDestroyStream(stream_ptr_);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[HWAscendNPU] destroy stream failed";
    }
    stream_ptr_ = nullptr;
  }
  LOG(INFO) << "[HWAscendNPU] end to destroy stream";

  if (context_ptr_ != nullptr) {
    ret = aclrtDestroyContext(context_ptr_);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "[HWAscendNPU] destroy context failed";
    }
    context_ptr_ = nullptr;
  }
  LOG(INFO) << "[HWAscendNPU] Release device successfully";
}

}  // namespace hw_ascend_npu
}  // namespace lite
}  // namespace paddle
