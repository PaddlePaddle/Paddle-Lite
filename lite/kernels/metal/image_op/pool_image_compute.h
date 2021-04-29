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

#ifndef LITE_KERNELS_METAL_IMAGE_OP_POOL_IMAGE_COMPUTE_H_
#define LITE_KERNELS_METAL_IMAGE_OP_POOL_IMAGE_COMPUTE_H_

#include <memory>
#include <string>

#include "lite/core/kernel.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

#include "lite/backends/metal/metal_debug.h"
#include "lite/backends/metal/metal_context.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

class PoolImageCompute : public KernelLite<TARGET(kMetal),
                                           PRECISION(kFloat),
                                           DATALAYOUT(kMetalTexture2DArray)> {
  using param_t = operators::PoolParam;

 public:
  void PrepareForRun() override;
  void Run() override;
  void SaveOutput() override {
    MetalDebug::SaveOutput("pool", output_buffer_);
  };
  virtual ~PoolImageCompute();

 private:
	bool use_mps_{false};
	void *mps_pool_op_{nullptr};
	void *mps_input_image_{nullptr};
	void *mps_output_image_{nullptr};

	void setup_with_mps();
	void setup_without_mps();
																						 
	void run_with_mps();
	void run_without_mps();
															
  const MetalImage* input_buffer_;
  MetalImage* output_buffer_;
  std::shared_ptr<MetalBuffer> params_buffer_;
													
  void* pipline_;
	std::string function_name_;
  MetalContext* metal_context_;
};

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#endif  // LITE_KERNELS_METAL_IMAGE_OP_POOL_IMAGE_COMPUTE_H_
