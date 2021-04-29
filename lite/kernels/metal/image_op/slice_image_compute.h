//
//  slice_image_compute.h
//  PaddleLiteiOS
//
//  Created by hxwc on 2021/4/13.
//
#ifndef LITE_KERNELS_METAL_IMAGE_OP_SLICE_IMAGE_COMPUTE_H_
#define LITE_KERNELS_METAL_IMAGE_OP_SLICE_IMAGE_COMPUTE_H_

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

class SliceImageCompute : public KernelLite<TARGET(kMetal),
																					 PRECISION(kFloat),
																					 DATALAYOUT(kMetalTexture2DArray)> {
	using param_t = operators::SliceParam;

 public:
	void PrepareForRun() override;
	void Run() override;
	void SaveOutput() override {
		MetalDebug::SaveOutput("slice", output_buffer_);
	};

 private:
	void setup_without_mps();

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

#endif  // LITE_KERNELS_METAL_IMAGE_OP_SLICE_IMAGE_COMPUTE_H_
