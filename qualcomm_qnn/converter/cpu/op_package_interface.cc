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

#include "driver/qualcomm_qnn/converter/cpu/op_package.h"
#include "CPU/QnnCpuOpPackage.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace cpu {

Qnn_ErrorHandle_t Initialize(
    QnnOpPackage_GlobalInfrastructure_t global_infrastructure) {
  if (CpuOpPackage::GetIsInitialized()) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  }

  auto op_pkg = CpuOpPackage::GetInstance();
  if (!op_pkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  op_pkg->SetPackageInfo("Custom.Cpu.OpPackage");
  CpuOpPackage::SetIsInitialized(true);

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t GetInfo(const QnnOpPackage_Info_t** info) {
  auto op_pkg = CpuOpPackage::GetInstance();
  if (!op_pkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  QNN_CHECK_STATUS(op_pkg->GetPackageInfo(info));

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ValidateOpConfig(Qnn_OpConfig_t op_config) {
  auto op_pkg = CpuOpPackage::GetInstance();
  if (!op_pkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  QNN_CHECK_STATUS(op_pkg->ValidateOpConfig(op_config));

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t CreateOpImpl(
    QnnOpPackage_GraphInfrastructure_t graph_infrastructure,
    QnnOpPackage_Node_t node,
    QnnOpPackage_OpImpl_t* op_impl_ptr) {
  auto op_pkg = CpuOpPackage::GetInstance();
  if (!op_pkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  auto op_node = reinterpret_cast<QnnCpuOpPackage_Node_t*>(node);
  auto ops = reinterpret_cast<QnnCpuOpPackage_OpImpl_t**>(op_impl_ptr);
  QNN_CHECK_STATUS(op_pkg->CreateOpImpl(graph_infrastructure, op_node, ops));

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t FreeOpImpl(QnnOpPackage_OpImpl_t op_impl) {
  auto op_pkg = CpuOpPackage::GetInstance();
  if (!op_pkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  auto ops = reinterpret_cast<QnnCpuOpPackage_OpImpl_t*>(op_impl);
  QNN_CHECK_STATUS(op_pkg->FreeOpImpl(ops));

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t Terminate() {
  CpuOpPackage::DestroyInstance();

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t LogInitialize(QnnLog_Callback_t callback,
                                QnnLog_Level_t max_log_level) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t LogSetLevel(QnnLog_Level_t max_log_level) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t LogTerminate() { return QNN_SUCCESS; }

#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) Qnn_ErrorHandle_t
CpuCustomOpPackage_interfaceProvider(QnnOpPackage_Interface_t* interface) {
  interface->interfaceVersion.major = 1;
  interface->interfaceVersion.minor = 4;
  interface->interfaceVersion.patch = 0;
  interface->v1_4.init = Initialize;
  interface->v1_4.terminate = Terminate;
  interface->v1_4.getInfo = GetInfo;
  interface->v1_4.validateOpConfig = ValidateOpConfig;
  interface->v1_4.createOpImpl = CreateOpImpl;
  interface->v1_4.freeOpImpl = FreeOpImpl;
  interface->v1_4.logInitialize = LogInitialize;
  interface->v1_4.logSetLevel = LogSetLevel;
  interface->v1_4.logTerminate = LogTerminate;
  return QNN_SUCCESS;
}
#ifdef __cplusplus
}
#endif

}  // namespace cpu
}  // namespace qualcomm_qnn
}  // namespace nnadapter
