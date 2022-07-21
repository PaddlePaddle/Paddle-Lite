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

#include "driver/qualcomm_qnn/converter/htp/op_package.h"

#include "HTP/QnnHtpCommon.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace htp {

std::mutex HtpOpPackage::s_mtx;
std::shared_ptr<HtpOpPackage> HtpOpPackage::s_op_package;
bool HtpOpPackage::s_is_initialized = false;

bool HtpOpPackage::GetIsInitialized() {
  std::lock_guard<std::mutex> locker(s_mtx);
  return s_is_initialized;
}

void HtpOpPackage::DestroyInstance() {
  SetIsInitialized(false);
  s_op_package.reset();
}

void HtpOpPackage::SetIsInitialized(bool is_initialized) {
  std::lock_guard<std::mutex> locker(s_mtx);
  s_is_initialized = is_initialized;
}

std::shared_ptr<HtpOpPackage> HtpOpPackage::GetInstance() {
  std::lock_guard<std::mutex> locker(s_mtx);
  if (!s_op_package) {
    s_op_package.reset(new (std::nothrow) HtpOpPackage());
  }

  return s_op_package;
}

Qnn_ErrorHandle_t HtpOpPackage::SetPackageInfo(const char* package_name) {
  std::vector<const char*> supported_custom_ops;
#define REGISTER_CUSTOM_OP(__op_type__) \
  supported_custom_ops.push_back(#__op_type__);
#include "driver/qualcomm_qnn/converter/htp/op_list.h"  // NOLINT
#undef __NNADAPTER_DRIVER_QUALCOMM_QNN_CONVERTER_HTP_OP_LIST_H__
#undef REGISTER_CUSTOM_OP
  package_name_.assign(package_name);
  sdk_api_version_ = QNN_HTP_API_VERSION_INIT;
  package_info_ = {
      package_name_.c_str(),
      supported_custom_ops.data(),                         // Operations
      nullptr,                                             // Operation info
      static_cast<uint32_t>(supported_custom_ops.size()),  // Num Operations
      nullptr,                                             // Optimizations
      0,                                                   // numOptimizations
      QNN_SDK_BUILD_ID,                                    // sdkBuildId
      &sdk_api_version_,                                   // sdkApiVersion
      nullptr,                                             // packageInfo
      {0}};                                                // reserved

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t HtpOpPackage::GetPackageInfo(
    const QnnOpPackage_Info_t** info) {
  *info = &package_info_;

  return QNN_SUCCESS;
}

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter
