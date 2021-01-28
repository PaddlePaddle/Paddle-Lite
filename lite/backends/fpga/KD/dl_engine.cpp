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

#include "lite/backends/fpga/KD/dl_engine.hpp"
#include "lite/core/version.h"

namespace paddle {
namespace zynqmp {

DLEngine::DLEngine() {
  open_device();
  int ret = get_device_info(info_);
  filter::set_filter_capacity(info_.filter_cap);
  filter::set_colunm(info_.column);

  char buff[21] = {0};
  struct VersionArgs args = {.buffer = buff, .size = 21};
  ret = get_version(args);

  std::string version = lite::paddlelite_branch();
  std::string commit_hash = lite::paddlelite_commit();

  if (ret == 0) {
    char* driver_version = reinterpret_cast<char*>(args.buffer);
    std::cout << "driver_version: " << std::string(driver_version) << std::endl;
  }
  std::cout << "paddle_lite_version: " << version << "-" << commit_hash
            << std::endl;
}

}  // namespace zynqmp
}  // namespace paddle
