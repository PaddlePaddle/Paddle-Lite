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
#include <algorithm>
#include <utility>
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {

Engine::Engine(KernelContext *ctx,
               int block_idx,
               cpp::BlockDesc *block_desc,
               const std::vector<std::string> &input_names,
               const std::vector<std::string> &output_names,
               lite::Scope *scope)
    : ctx_(ctx), block_idx_(block_idx), block_desc_(block_desc), scope_(scope) {
  input_names_ = input_names;
  output_names_ = output_names;
  // Sort the name of input and output tensors, it's convenient for us to get
  // the info of input and output tensors in the same order from the device
  // program, because the result of subgraph division may be different but right
  // at each call of the subgraph pass.
  std::stable_sort(input_names_.begin(), input_names_.end());
  std::stable_sort(output_names_.begin(), output_names_.end());
}

bool Engine::Run() {
  if (is_first_epoch_) {
    PrepareWorkspaceForDeviceProgram();
    is_first_epoch_ = false;
  }
  if (InputShapeChanged()) {
    BuildDeviceProgram();
  }
  return LaunchDeviceProgram();
}

bool Engine::PrepareWorkspaceForOriginProgram() {
  origin_idims_.resize(input_names_.size());
  origin_itensors_.resize(input_names_.size());
  for (int i = 0; i < input_names_.size(); i++) {
    origin_itensors_[i] = scope_->FindMutableTensor(input_names_[i]);
    CHECK(origin_itensors_[i]);
  }
  origin_otensors_.resize(output_names_.size());
  for (int i = 0; i < output_names_.size(); i++) {
    origin_otensors_[i] = scope_->FindMutableTensor(output_names_[i]);
    CHECK(origin_otensors_[i]);
  }
  return true;
}

bool Engine::BuildOriginProgram() {
  // TODO(hong19860320) The block_desc need to be divided into subgraphs during
  // the exection time. But only see them as a subgraph now.
  origin_program_.clear();
  for (size_t op_idx = 0; op_idx < block_desc_->OpsSize(); op_idx++) {
    auto op_desc = block_desc_->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    // Create op and pick up the best kernel
    auto op = LiteOpRegistry::Global().Create(op_desc->Type());
    CHECK(op) << "no Op found for " << op_type;
    op->Attach(*op_desc, scope_);
    std::unique_ptr<KernelBase> picked_kernel;
    if (op_desc->HasAttr(kKernelTypeAttr)) {
      // Create op and pick up the best kernel according to the
      // kKernelTypeAttr attribute
      auto kernel_type = op_desc->GetAttr<std::string>(kKernelTypeAttr);
      std::string alias;
      Place place;
      KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      VLOG(3) << "Found the attr '" << kKernelTypeAttr << "': " << kernel_type
              << " for " << op_type;
      auto kernels = op->CreateKernels({place});
      CHECK_GT(kernels.size(), 0u) << "No kernels found for " << op_type;
      auto it = std::find_if(
          kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase> &it) {
            return it->alias() == alias;
          });
      CHECK(it != kernels.end());
      picked_kernel = std::move(*it);
    } else {
      // TODO(hong19860320) add kernel picking according to the type of input
      // and output tensors
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
  CHECK(!origin_program_.empty()) << "no instructions";
  return true;
}

bool Engine::LaunchOriginProgram() {
  if (origin_program_.empty()) {
    BuildOriginProgram();
  }
  if (!origin_program_.empty()) {
    for (auto &inst : origin_program_) {
      auto op_type = inst.op()->op_info()->Type();
      if (op_type == "feed" || op_type == "fetch") continue;
      inst.Run();
    }
    return true;
  }
  return false;
}

bool Engine::PrepareWorkspaceForDeviceProgram() {
  return PrepareWorkspaceForOriginProgram();
}

bool Engine::BuildDeviceProgram() { return BuildOriginProgram(); }

bool Engine::LaunchDeviceProgram() { return LaunchOriginProgram(); }

bool Engine::InputShapeChanged() {
  bool changed = false;
  for (size_t i = 0; i < origin_itensors_.size(); i++) {
    auto origin_idim = origin_itensors_[i]->dims().Vectorize();
    changed |= origin_idim != origin_idims_[i];
    origin_idims_[i] = origin_idim;
  }
  return changed;
}

}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
