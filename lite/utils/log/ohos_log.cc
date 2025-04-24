// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(__OHOS__) && defined(LITE_WITH_LOG)
#include "lite/utils/log/ohos_log.h"
#include "hilog/log.h"
namespace ohos {
//  OHOS_LOG_I("device_name %s", device_name.c_str());
#undef LOG_INFO
#undef LOG_WARN

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200           // global domain
#define LOG_TAG "Paddle-Lite-OHOS"  // global tag

#define OHOS_LOG_I(format, ...) \
  OH_LOG_INFO(LogType::LOG_APP, "【native】Info: " format, ##__VA_ARGS__);

#define OHOS_LOG_W(format, ...) \
  OH_LOG_WARN(LogType::LOG_APP, "【native】Info: " format, ##__VA_ARGS__);

void log(std::string& info) {
  OHOS_LOG_I("device_name %{public}s", info.c_str());
}
void log(const std::string& info) {
  OHOS_LOG_I("device_name %{public}s", info.c_str());
}

#undef LOG_DOMAIN
#undef LOG_TAG
}
#endif
