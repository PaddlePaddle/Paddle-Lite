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
#include "lite/operators/conditional_block_op.h"
#include "lite/operators/subgraph_op.h"
#include "lite/operators/while_op.h"

namespace paddle {
namespace lite {
namespace subgraph {

Engine::Engine(KernelContext *ctx,
               int block_idx,
               cpp::ProgramDesc *program_desc,
               const std::vector<std::string> &input_names,
               const std::vector<std::string> &output_names,
               const std::vector<std::string> &cached_shapes,
               lite::Scope *scope,
               std::string model_cache_dir)
    : ctx_(ctx),
      block_idx_(block_idx),
      program_desc_(program_desc),
      input_names_(input_names),
      output_names_(output_names),
      scope_(scope),
      model_cache_dir_(model_cache_dir) {
  for (auto &cached_shape : cached_shapes) {
    /*
    auto cached_input_output_shape = Split<std::string>(cached_shape, " ");
    CHECK(cached_input_output_shape.size() >= 1 &&
          cached_input_output_shape.size() <= 2);
    // parsing input data shape
    std::vector<Shape> cached_input_shapes;
    if (cached_input_output_shape.size() >= 1) {
      auto cached_input_shapes =
    Split<std::string>(cached_input_output_shape[0], ";"); for (auto& i :
    cached_input_shapes) { auto cached_input_shape = Split<std::string>(i, "-");
        auto cached_input_dims = cached_input_shape[0];
        if (cached_input_shape.size() >= 1) {

        }
        auto cached_input_lod =
      }
      */
    LOG(INFO) << cached_shape;
  }
}

int Engine::BuildDeviceProgram() { return FAILED; }

int Engine::LaunchDeviceProgram() { return 0; }

int Engine::BuildOriginProgram() {
  // TODO(hong19860320) The block_desc need to be divided into subgraphs during
  // the exection time. But only see them as a subgraph now.
  origin_program_.clear();
  CHECK(block_idx_ >= 0 && block_idx_ < program_desc_->BlocksSize());
  auto *block_desc = program_desc_->GetBlock<cpp::BlockDesc>(block_idx_);
  for (size_t op_idx = 0; op_idx < block_desc->OpsSize(); op_idx++) {
    auto *op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    auto op = LiteOpRegistry::Global().Create(op_desc->Type());
    CHECK(op) << "no Op found for " << op_type;
    if (op_type == "while") {
      static_cast<operators::WhileOpLite *>(op.get())->SetProgramDesc(
          program_desc);
    } else if (op_type == "conditional_block") {
      static_cast<operators::ConditionalBlockOpLite *>(op.get())
          ->SetProgramDesc(program_desc);
    } else if (op_type == "subgraph") {
      static_cast<operators::SubgraphOp *>(op.get())->SetProgramDesc(
          program_desc);
    }
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
      CHECK_GT(kernels.size(), 0u) << "No kernels found for " << op_type;
      auto it = std::find_if(
          kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase> &it) {
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
  for (auto &inst : origin_program_) {
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

void Engine::InitDeviceTensor() { return; }

bool Engine::InputShapeChanged() {
  for (size_t i = 0; i < origin_itensors_.size(); i++) {
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
    InitDeviceTensor();
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
