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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/cxx_api.h"
#include "lite/api/test_helper.h"

namespace paddle {
namespace lite {

void TestModel(const std::vector<Place>& valid_places,
               const std::string& model_dir = FLAGS_model_dir,
               bool save_model = false) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);
  DeviceInfo::Global().ExtendWorkspace(sizeof(float) * 524288);
}

}  // namespace lite
}  // namespace paddle
