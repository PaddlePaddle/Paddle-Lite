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

#include "lite/core/subgraph_engine_base.h"
#include <sys/time.h>
#include <time.h>
#include <algorithm>
#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {

SubgraphEngineBase::SubgraphEngineBase(
    KernelContext *ctx,
    int block_idx,
    const std::shared_ptr<const cpp::ProgramDesc> &program_desc,
    Scope *exec_scope,
    const std::vector<std::string> &input_names,
    const std::vector<std::string> &output_names)
    : ctx_(ctx),
      block_idx_(block_idx),
      program_desc_(program_desc),
      exec_scope_(exec_scope) {
  input_names_ = input_names;
  output_names_ = output_names;
  // Sort the name of input and output tensors, it's convenient for us to get
  // the info of input and output tensors in the same order from the device
  // program, because the result of subgraph division may be different but right
  // at each call of the subgraph pass.
  std::stable_sort(input_names_.begin(), input_names_.end());
  std::stable_sort(output_names_.begin(), output_names_.end());
}

bool SubgraphEngineBase::Run() {
  if (is_first_epoch_) {
    PrepareWorkspaceForDeviceProgram();
    is_first_epoch_ = false;
  }
  if (InputShapeChanged()) {
    BuildDeviceProgram();
  }
  return LaunchDeviceProgram();
}

bool SubgraphEngineBase::PrepareWorkspaceForOriginProgram() {
  origin_idims_.resize(input_names_.size());
  origin_itensors_.resize(input_names_.size());
  for (int i = 0; i < input_names_.size(); i++) {
    origin_itensors_[i] = exec_scope_->FindMutableTensor(input_names_[i]);
    CHECK(origin_itensors_[i]);
  }
  origin_otensors_.resize(output_names_.size());
  for (int i = 0; i < output_names_.size(); i++) {
    origin_otensors_[i] = exec_scope_->FindMutableTensor(output_names_[i]);
    CHECK(origin_otensors_[i]);
  }
  return true;
}

bool SubgraphEngineBase::BuildOriginProgram() {
  // TODO(hong19860320) The block_desc need to be divided into subgraphs during
  // the exection time. But only see them as a subgraph now.
  if (!origin_program_) {
    origin_program_.reset(
        new RuntimeProgram(program_desc_, exec_scope_, block_idx_));
  }
  return true;
}

bool SubgraphEngineBase::LaunchOriginProgram() {
  if (!origin_program_) {
    BuildOriginProgram();
  }
  if (origin_program_) {
    VLOG(3) << "Roll back to run the origin program.";
    origin_program_->Run();
    return true;
  }
  return false;
}

bool SubgraphEngineBase::PrepareWorkspaceForDeviceProgram() {
  return PrepareWorkspaceForOriginProgram();
}

bool SubgraphEngineBase::BuildDeviceProgram() { return BuildOriginProgram(); }

bool SubgraphEngineBase::LaunchDeviceProgram() { return LaunchOriginProgram(); }

bool SubgraphEngineBase::InputShapeChanged() {
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
