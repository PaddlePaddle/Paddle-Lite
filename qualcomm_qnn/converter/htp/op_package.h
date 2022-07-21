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

#include <cstring>
#include <memory>
#include <mutex>  //NOLINT
#include <string>
#include <vector>

#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"
#include "QnnTypes.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace htp {

class HtpOpPackage {
 public:
  static std::shared_ptr<HtpOpPackage> GetInstance();

  static void DestroyInstance();

  static bool GetIsInitialized();

  static void SetIsInitialized(bool is_initialized);

  Qnn_ErrorHandle_t SetPackageInfo(const char* package_name);

  Qnn_ErrorHandle_t GetPackageInfo(const QnnOpPackage_Info_t** info);

  ~HtpOpPackage() { DestroyInstance(); }

 private:
  HtpOpPackage() = default;
  HtpOpPackage(const HtpOpPackage& other) = delete;
  HtpOpPackage& operator=(const HtpOpPackage& other) = delete;

  std::string package_name_;
  QnnOpPackage_Info_t package_info_;
  Qnn_ApiVersion_t sdk_api_version_;

  // static
  static std::mutex s_mtx;
  static std::shared_ptr<HtpOpPackage> s_op_package;
  static bool s_is_initialized;
};

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter
