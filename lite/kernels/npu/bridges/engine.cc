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
               Scope *exec_scope,
               const std::vector<std::string> &input_names,
               const std::vector<std::string> &output_names,
               const std::vector<std::string> &cached_shapes,
               std::string model_cache_dir)
    : ctx_(ctx),
      block_idx_(block_idx),
      program_desc_(program_desc),
      exec_scope_(exec_scope),
      input_names_(input_names),
      output_names_(output_names),
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
  origin_program_.reset(
      new RuntimeProgram(block_idx_, program_desc_, exec_scope_));
  return 0;
}

int Engine::LaunchOriginProgram() {
  origin_program_->Run();
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
