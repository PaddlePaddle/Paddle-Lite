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

#pragma once

#include <cstdarg>
#include <string>
#include <vector>
#include "lite/core/device_info.h"
#include "lite/core/tensor.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_BM
template <>
class Device<TARGET(kBM)> {
 public:
  Device(int dev_id, int max_stream = 1)
      : idx_(dev_id), max_stream_(max_stream) {}
  void Init();

  int id() { return idx_; }
  int max_stream() { return 1; }
  std::string name() { return "BM"; }
  float max_memory() { return 16; }
  int core_num();
  void SetId(int idx);

  int sm_version() { return 0; }
  bool has_fp16() { return false; }
  bool has_int8() { return false; }
  bool has_hmma() { return false; }
  bool has_imma() { return false; }
  int runtime_version() { return 0; }

 private:
  void CreateQueue() {}
  void GetInfo() {}

 private:
  int idx_{0};
  int max_stream_{1};
  std::string device_name_;
  float max_memory_;

  int sm_version_;
  bool has_fp16_;
  bool has_int8_;
  bool has_hmma_;
  bool has_imma_;
  int runtime_version_;
};

template class Env<TARGET(kBM)>;
#endif
}  // namespace lite
}  // namespace paddle
