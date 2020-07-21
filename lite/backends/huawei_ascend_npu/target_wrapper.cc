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

#include "lite/backends/huawei_ascend_npu/target_wrapper.h"
#include <algorithm>
#include <map>
#include <string>

namespace paddle {
namespace lite {

std::mutex TargetWrapperHuaweiAscendNPU::device_mutex_;
bool TargetWrapperHuaweiAscendNPU::runtime_inited_{false};
std::vector<int> TargetWrapperHuaweiAscendNPU::device_list_{};

thread_local int TargetWrapperHuaweiAscendNPU::device_id_{0};

void TargetWrapperHuaweiAscendNPU::InitOnce() {
  if (runtime_inited_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] runtime already inited!";
    return;
  }
  // ACL runtime init => can only be called once in one process
  ACL_CALL(aclInit(NULL));

  // ATC builder init => can only be called once in one process
  std::map<std::string, std::string> global_options;
  global_options.insert(
      std::make_pair(ge::ir_option::SOC_VERSION, "Ascend310"));
  ATC_CALL(ge::aclgrphBuildInitialize(global_options));

  runtime_inited_ = true;
}

void TargetWrapperHuaweiAscendNPU::DestroyOnce() {
  if (!runtime_inited_) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] no need to destroy runtime!";
    return;
  }
  // ATC builder finalize => can only be called once in one process
  ge::aclgrphBuildFinalize();
  // ACL runtime finalize => can only be called once in one process
  ACL_CALL(aclFinalize());

  runtime_inited_ = false;
}

size_t TargetWrapperHuaweiAscendNPU::num_devices() {
  uint32_t count = 0;
  ACL_CALL(aclrtGetDeviceCount(&count));
  return count;
}

void TargetWrapperHuaweiAscendNPU::CreateDevice(int device_id) {
  std::lock_guard<std::mutex> lock(device_mutex_);

  if (!runtime_inited_) {
    VLOG(3)
        << "[HUAWEI_ASCEND_NPU] Acl runtime not initialized, start to init ...";
    InitOnce();
  }

  VLOG(3) << "[HUAWEI_ASCEND_NPU] Setting Huawei Ascend Device to "
          << device_id;
  if (device_id < 0 || device_id >= num_devices()) {
    LOG(FATAL) << "Failed with invalid device id " << device_id;
    return;
  }
  if (std::find(device_list_.begin(), device_list_.end(), device_id) !=
      device_list_.end()) {
    LOG(WARNING) << "Device already in use, device id " << device_id;
    return;
  }
  device_id_ = device_id;

  ACL_CALL(aclrtSetDevice(device_id_));

  device_list_.push_back(device_id_);
}

void TargetWrapperHuaweiAscendNPU::DestroyDevice(int device_id) {
  std::lock_guard<std::mutex> lock(device_mutex_);

  VLOG(3) << "[HUAWEI_ASCEND_NPU] Release Huawei Ascend Device ID: "
          << device_id;
  auto iter = std::find(device_list_.begin(), device_list_.end(), device_id);
  if (iter == device_list_.end()) {
    LOG(WARNING) << "Device not in use, device id " << device_id;
    return;
  }
  CHECK_EQ(device_id, device_id_)
      << "Releasing deivce id not equals to current thread local device id!!!";

  ACL_CALL(aclrtResetDevice(device_id_));

  device_list_.erase(iter);

  if (device_list_.empty()) {
    VLOG(3) << "[HUAWEI_ASCEND_NPU] No active session, destroy acl rumtime and "
               "atc builder ...";
    DestroyOnce();
  }
}

}  // namespace lite
}  // namespace paddle
