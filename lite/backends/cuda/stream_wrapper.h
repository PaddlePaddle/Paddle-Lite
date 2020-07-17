// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_runtime.h>
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/target_wrapper.h"

namespace paddle {
namespace lite {

class StreamWrapper {
 public:
  explicit StreamWrapper(cudaStream_t stream)
      : stream_(stream), owner_(false) {}
  StreamWrapper() : owner_(true) {
    lite::TargetWrapperCuda::CreateStream(&stream_);
  }
  ~StreamWrapper() {
    if (owner_) {
      lite::TargetWrapperCuda::DestroyStream(stream_);
    }
  }
  cudaStream_t stream() { return stream_; }
  bool owner() { return owner_; }

 private:
  cudaStream_t stream_;
  bool owner_;
};

}  // namespace lite
}  // namespace paddle
