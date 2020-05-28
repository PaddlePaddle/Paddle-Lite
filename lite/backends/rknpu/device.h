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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "rknpu/rknpu_pub.h"  // NOLINT

namespace paddle {
namespace lite {
namespace rknpu {

class Device {
 public:
  static Device& Global() {
    static Device x;
    return x;
  }
  Device() {}

  // Build the RK IR graph to om model, return RK model exector to
  // load om model and run inference.
  std::unique_ptr<rk::nn::Exection> Build(
      std::string& model_name,                                   // NOLINT
      rk::nn::Graph* rk_graph,                                   // NOLINT
      std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes,  // NOLINT
      std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes  // NOLINT
      );                                                         // NOLINT

 private:
};

}  // namespace rknpu
}  // namespace lite
}  // namespace paddle
