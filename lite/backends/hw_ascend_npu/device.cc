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

#include "lite/backends/hw_ascend_npu/device.h"
#include <map>
#include <string>
#include "ge/ge_api_types.h"
#include "lite/backends/hw_ascend_npu/runtime.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace hw_ascend_npu {
std::shared_ptr<HWAscendNPURuntime> Device::Build(
    std::vector<ge::Operator>& input_nodes,  // NOLINT
    std::vector<ge::Operator>& output_nodes  // NOLINT
    ) {
  VLOG(3) << "[HWAscendNPU] Build model";
  // Build the IR graph to the om model
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  ge::ModelBufferData model;

  std::map<std::string, std::string> build_options;
  build_options.insert({ge::ir_option::EXEC_DISABLE_REUSED_MEMORY, "1"});

  ge::graphStatus ret = aclgrphBuildModel(ir_graph, build_options, model);

  if (ret != ge::GRAPH_SUCCESS) {
    LOG(ERROR) << "[HWAscendNPU] Build model failed, error code: " << ret;
    return nullptr;
  }

  std::shared_ptr<HWAscendNPURuntime> model_runtime(
      new HWAscendNPURuntime(model.data, model.length));
  CHECK(model_runtime != nullptr);
  if (!model_runtime->model_loaded()) {
    LOG(ERROR) << "[HWAscendNPU]: Can not create model runtime instance";
    return nullptr;
  }
  VLOG(3) << "[HWAscendNPU]: Build done";
  return model_runtime;
}

}  // namespace hw_ascend_npu
}  // namespace lite
}  // namespace paddle
