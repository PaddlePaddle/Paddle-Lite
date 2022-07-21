// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <memory>
#include <mutex>  //NOLINT
#include <string>
#include <unordered_map>
#include <vector>

#include "CPU/QnnCpuOpPackage.h"
#include "QnnTypes.h"

#include "driver/qualcomm_qnn/converter/cpu/op_base.h"
#include "driver/qualcomm_qnn/converter/cpu/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace cpu {

class CpuOpPackage {
 public:
  static std::shared_ptr<CpuOpPackage> GetInstance();

  static void DestroyInstance();

  static bool GetIsInitialized();

  static void SetIsInitialized(bool is_initialized);

  Qnn_ErrorHandle_t SetPackageInfo(const char* package_name);

  Qnn_ErrorHandle_t GetPackageInfo(const QnnOpPackage_Info_t** info);

  std::shared_ptr<OpBase> GetObject(size_t handle) {
    std::shared_ptr<OpBase> op;
    std::lock_guard<std::mutex> locker(s_mtx);
    if (op_map_.find(handle) == op_map_.end()) {
      return op;
    }
    op = op_map_.find(handle)->second;

    return op;
  }

  Qnn_ErrorHandle_t RemoveObject(size_t handle) {
    std::lock_guard<std::mutex> locker(s_mtx);
    if (op_map_.find(handle) != op_map_.end()) {
      op_map_.erase(handle);
    }

    return QNN_SUCCESS;
  }

  size_t GetHandle(std::shared_ptr<OpBase> op) {
    std::lock_guard<std::mutex> locker(s_mtx);
    size_t handle = size_t(op.get());
    op_map_[handle] = op;

    return handle;
  }

  Qnn_ErrorHandle_t CreateOpImpl(
      QnnCpuOpPackage_GraphInfrastructure_t* graph_infrastructure,
      QnnCpuOpPackage_Node_t* node,
      QnnCpuOpPackage_OpImpl_t** op_impl_ptr);

  Qnn_ErrorHandle_t ValidateOpConfig(Qnn_OpConfig_t op_config) {
    (void)op_config;

    return QNN_SUCCESS;
  }

  Qnn_ErrorHandle_t ExecuteNode(void* kernel_handle);

  Qnn_ErrorHandle_t FreeOpImpl(QnnCpuOpPackage_OpImpl_t* op_impl);

  ~CpuOpPackage() { DestroyInstance(); }

 private:
  CpuOpPackage() = default;
  CpuOpPackage(const CpuOpPackage& other) = delete;
  CpuOpPackage& operator=(const CpuOpPackage& other) = delete;

  std::string package_name_;
  std::unordered_map<size_t, std::shared_ptr<OpBase>> op_map_;
  QnnOpPackage_Info_t package_info_;
  Qnn_ApiVersion_t sdk_api_version_;

  // static
  static std::mutex s_mtx;
  static std::shared_ptr<CpuOpPackage> s_op_package;
  static bool s_is_initialized;
};

}  // namespace cpu
}  // namespace qualcomm_qnn
}  // namespace nnadapter
