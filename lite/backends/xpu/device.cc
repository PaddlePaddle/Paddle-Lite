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
#include "lite/utils/cp_logging.h"

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

  // The XPU compiler build the graph and fill all of the constant params, only
  // one output is supported now.
  xtcl::xNetwork network = builder->FinalizeNetwork(*((*outputs)[0]));
  auto target = xtcl::Target::Create(device_name_);
  auto compiler = xtcl::network::xTensorCompiler(network, target);
  compiler.SetParams(*params);  // Set the data of constant tensors
  compiler.Build();
  VLOG(3) << "[NPU] Build done";
  return std::unique_ptr<xtcl::network::xRuntimeInstance>(
      new xtcl::network::xRuntimeInstance(compiler.CreateRuntimeInstance()));
}

}  // namespace xpu
}  // namespace lite
}  // namespace paddle
