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

#include "lite/backends/xpu/device.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace xpu {

std::unique_ptr<xtcl::network::xRuntimeInstance> Device::Build(
    xtcl::network::xNetworkBuilder* builder,
    xtcl::network::xTensorCompiler::ParamNDArrayMap* params,
    std::vector<xtcl::xExpr*>* outputs) {
  VLOG(3) << "[XPU] Build model";
  CHECK(builder != nullptr);
  CHECK(outputs != nullptr);
  CHECK_GT(outputs->size(), 0);

  // The XPU compiler build the graph and fill all of the constant params, and
  // use TupleNode to support multiple outputs
  xtcl::Array<xtcl::xExpr> all_outs;
  for (size_t i = 0; i < outputs->size(); i++) {
    all_outs.push_back(*outputs->at(i));
  }
  xtcl::xFunction network =
      builder->FinalizeNetwork(xtcl::relay::Tuple(all_outs));
  auto target = xtcl::NullValue<xtcl::Target>();
  if (!target_.empty()) {
    target = xtcl::Target(target_);
  }
  xtcl::network::xTensorCompiler compiler(network, target);
  compiler.SetParams(*params);  // Set the data of constant tensors
  compiler.Build();
  VLOG(3) << "[XPU] Build done";

  int device_id = 0;
  auto device_str = std::getenv("XPU_VISIBLE_DEVICES");
  if (device_str && atoi(device_str)) device_id = atoi(device_str);
  VLOG(3) << "[XPU] device id: " << device_id;

  return std::unique_ptr<xtcl::network::xRuntimeInstance>(
      compiler.CreateRuntimeInstancePtr(device_id));
}

}  // namespace xpu
}  // namespace lite
}  // namespace paddle
