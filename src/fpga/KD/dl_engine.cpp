/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "dl_engine.hpp"
namespace paddle_mobile {
namespace zynqmp {

DeviceInfo& DLEngine::deviceInfo() { return info_; }

DLEngine::DLEngine() {
  open_device();
  int ret = get_device_info(info_);
  filter::set_filter_capacity(2048);
  // filter::set_filter_capacity(info_.filter_cap);

  // std::cout << "version:" << info_.version;
  // std::cout << "device_type:" << info_.device_type;
  // std::cout << "filter_cap:" << info_.filter_cap;
}

}  // namespace zynqmp
}  // namespace paddle_mobile
