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

#include "lite/kernels/npu/bridges/engine.h"
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {

int Engine::BuildDeviceProgram() { return FAILED; }

int Engine::LaunchDeviceProgram() { return 0; }

int Engine::BuildOriginProgram() {
  // TODO(hong19860320) The block_desc need to be divided into subgraphs during
  // the exection time. But only see them as a subgraph now.
  origin_program_.clear();
  for (int op_idx = 0; op_idx < block_desc_->OpsSize(); op_idx++) {
    auto op_desc = block_desc_->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    auto op = LiteOpRegistry::Global().Create(op_desc->Type());
    op->Attach(*op_desc, scope_);
    std::unique_ptr<KernelBase> picked_kernel;
    if (op_desc->HasAttr(kKernelTypeAttr)) {
      // Create op and pick up kernel according to the kKernelTypeAttr attribute
      auto kernel_type = op_desc->GetAttr<std::string>(kKernelTypeAttr);
      std::string alias;
      Place place;
      KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      VLOG(3) << "Found the attr '" << kKernelTypeAttr << "': " << kernel_type
              << " for " << op_type;
      auto kernels = op->CreateKernels({place});
      CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
      auto it = std::find_if(
          kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase>& it) {
            return it->alias() == alias;
          });
      CHECK(it != kernels.end());
      picked_kernel = std::move(*it);
    } else {
      VLOG(3) << "The attr '" << kKernelTypeAttr
              << "' not found, pick the first kernel for " << op_type;
      std::vector<std::unique_ptr<KernelBase>> kernels;
#if defined(LITE_WITH_ARM)
      kernels = op->CreateKernels({Place{TARGET(kARM)}, Place{TARGET(kHost)}});
#elif defined(LITE_WITH_X86)
      kernels = op->CreateKernels({Place{TARGET(kX86)}, Place{TARGET(kHost)}});
#endif
      if (kernels.size() > 0) {
        picked_kernel = std::move(kernels.front());
      } else {
        LOG(WARNING) << "No kernels found for " << op_type;
      }
    }
    if (picked_kernel != nullptr) {
      picked_kernel->SetContext(
          ContextScheduler::Global().NewContext(picked_kernel->target()));
    }
    origin_program_.emplace_back(std::move(op), std::move(picked_kernel));
  }
  return 0;
}

int Engine::LaunchOriginProgram() {
  for (auto& inst : origin_program_) {
    auto op_type = inst.op()->op_info()->Type();
    if (op_type == "feed" || op_type == "fetch") continue;
    inst.Run();
  }
  return 0;
}

int Engine::Build() {
  // In order to attach all of the ops of the block desc, we need to build the
  // original program firstly.
  BuildOriginProgram();
  // Run InferShape() of all of ops, and convert Paddle ops to NPU/XPU IR graph
  build_device_program_status_ = BuildDeviceProgram();
  return build_device_program_status_;
}

bool Engine::InputShapeChanged() {
  for (int i = 0; i < origin_itensors_.size(); i++) {
    if (origin_itensors_[i]->dims() != origin_idims_[i]) {
      return true;
    }
  }
  return false;
}

int Engine::Launch() {
  // Rebuild device program when the shapes of input tensors have been changed.
  if (CHECK_SUCCESS(build_device_program_status_) &&
      CHECK_REBUILD_WHEN_SHAPE_CHANGED(build_device_program_status_) &&
      InputShapeChanged()) {
    Build();
  }
  if (CHECK_FAILED(build_device_program_status_)) {
    LaunchOriginProgram();
  } else {
    LaunchDeviceProgram();
  }
  return 0;
}

}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
