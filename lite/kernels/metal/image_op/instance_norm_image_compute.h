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

#include <memory>

#include "lite/core/kernel.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

#include "lite/backends/metal/metal_context.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

class InstanceNormImageCompute
    : public KernelLite<TARGET(kMetal),
                        PRECISION(kFloat),
                        DATALAYOUT(kMetalTexture2DArray)> {
  using param_t = operators::InstanceNormParam;

 public:
  void PrepareForRun() override;
  void Run() override;

 private:
  const MetalImage* input_buffer_;
  MetalImage* output_buffer_;
  std::shared_ptr<MetalBuffer> params_buffer_;

  int input_tensor_n_{-1};
  int input_tensor_c_{-1};
  int input_tensor_h_{-1};
  int input_tensor_w_{-1};

  int output_tensor_n_{-1};
  int output_tensor_c_{-1};
  int output_tensor_h_{-1};
  int output_tensor_w_{-1};

  std::shared_ptr<MetalBuffer> bias_buffer_;
  std::shared_ptr<MetalBuffer> mean_buffer_;
  std::shared_ptr<MetalBuffer> scale_buffer_;
  std::shared_ptr<MetalBuffer> variance_buffer_;

  float epsilon_;
  float momentum;
};

class InstanceNormImageComputeHalf
    : public KernelLite<TARGET(kMetal),
                        PRECISION(kFP16),
                        DATALAYOUT(kMetalTexture2DArray)> {
  using param_t = operators::InstanceNormParam;

 public:
  void PrepareForRun() override;
  void Run() override;

 private:
  const MetalImage* input_buffer_;
  MetalImage* output_buffer_;
  int input_tensor_n_{-1};
  int input_tensor_c_{-1};
  int input_tensor_h_{-1};
  int input_tensor_w_{-1};
  int output_tensor_n_{-1};
  int output_tensor_c_{-1};
  int output_tensor_h_{-1};
  int output_tensor_w_{-1};
  std::shared_ptr<MetalBuffer> bias_buffer_;
  std::shared_ptr<MetalBuffer> mean_buffer_;
  std::shared_ptr<MetalBuffer> scale_buffer_;
  std::shared_ptr<MetalBuffer> variance_buffer_;
  std::shared_ptr<MetalBuffer> params_buffer_;
  float epsilon_;
  float momentum;
};

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
