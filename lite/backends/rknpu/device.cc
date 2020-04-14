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

#include "lite/backends/rknpu/device.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace rknpu {

std::unique_ptr<rk::nn::Exection> Device::Build(
    std::string& model_name,                                   // NOLINT
    rk::nn::Graph* rk_graph,                                   // NOLINT
    std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes,  // NOLINT
    std::vector<std::shared_ptr<rk::nn::Tensor>> output_nodes  // NOLINT
    ) {
  VLOG(3) << "[RKNPU] Build model";

  rk_graph->SetInputsOutputs(input_nodes, output_nodes);

  std::unique_ptr<rk::nn::Exection> exector =
      std::unique_ptr<rk::nn::Exection>(new rk::nn::Exection(rk_graph));

  exector->Build();

  return exector;
}

}  // namespace rknpu
}  // namespace lite
}  // namespace paddle
