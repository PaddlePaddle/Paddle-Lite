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

#include "lite/kernels/arm_trustzone/softmax_compute.h"
#include "lite/kernels/arm_trustzone/tee.h"
#include "lite/backends/arm/math/funcs.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif
#include "lite/core/tensor.h"
#include "lite/core/program.h"
#include <iomanip>
#include <stdlib.h>

namespace paddle {
namespace lite {
namespace kernels {
namespace arm_trustzone {

template <>
void SoftmaxCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = Param<operators::SoftmaxParam>();
  const float* din = param.x->data<float>();
  float* dout = param.output->mutable_data<float>();
  auto x_dims = param.x->dims();
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int axis_size = x_dims[axis];

  VLOG(4) << "Prepare to run softmax op in TEE...";

  CHECK(g_init_tee_context) << "TEE context is not initialized";

  VLOG(4) << "outer " << outer_num;
  VLOG(4) << "inner " << inner_num;
  VLOG(4) << "axis_size " << axis_size;

  // invoke TEE
  PT_SoftmaxParam pt_softmax_param;
  pt_softmax_param.x = convert_to_portable_tensor(param.x, PT_DataType::kPTFloat, false);
  pt_softmax_param.output = convert_to_portable_tensor(param.output, PT_DataType::kPTFloat, true);
  pt_softmax_param.axis = axis;
  pt_softmax_param.use_cudnn = false;

  handle_t param_handle = create_tee_param(SupportedOp::Softmax, (void*)&pt_softmax_param);
  VLOG(4) << "Get handle:" << param_handle;
  if (tee_run(SupportedOp::Softmax, param_handle) != 0) {
    VLOG(4) << "TEE run error";
    return;
  }

  VLOG(4) << "TEE run finished, then fetch output tensor from TEE:";
  // write output in CA and free the handle
  if (fetch_output_tensor(SupportedOp::Softmax, param_handle, pt_softmax_param.output) != 0) {
    VLOG(4) << "fetch_output_tensor error";
    return;
  }
  VLOG(4) << "Fetch output tensor finished";

  //delete dl_param;
}

#ifdef ENABLE_ARM_FP16
template <>
void SoftmaxCompute<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = Param<operators::SoftmaxParam>();
  const float16_t* din = param.x->data<float16_t>();
  float16_t* dout = param.output->mutable_data<float16_t>();
  auto x_dims = param.x->dims();
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int axis_size = x_dims[axis];
  if (inner_num == 1) {
    if (axis_size >= 8) {
      lite::arm::math::fp16::softmax_inner1_large_axis_fp16(
          din, dout, outer_num, axis_size);
    } else {
      lite::arm::math::fp16::softmax_inner1_small_axis_fp16(
          din, dout, outer_num, axis_size);
    }
  } else {
    int compute_size = outer_num * inner_num;
    if (axis_size == 4 && inner_num % 8 == 0) {
      lite::arm::math::fp16::softmax_inner8_axis4_fp16(
          din, dout, axis_size, inner_num, outer_num);
    } else {
      if (inner_num % 8 == 0) {
        lite::arm::math::fp16::softmax_inner8_axis1_fp16(
            din, dout, axis_size, inner_num, outer_num);
      } else {
        lite::arm::math::fp16::softmax_basic_fp16(
            din, dout, axis_size, inner_num, outer_num);
      }
    }
  }
}
#endif  // ENABLE_ARM_FP16
}  // namespace arm_trustzone
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm_trustzone::SoftmaxCompute<PRECISION(kFP16),
                                                   PRECISION(kFP16)>
    SoftmaxFp16;
REGISTER_LITE_KERNEL(softmax, kARMTrustZone, kFP16, kNCHW, SoftmaxFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARMTrustZone), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

typedef paddle::lite::kernels::arm_trustzone::SoftmaxCompute<PRECISION(kFloat),
                                                   PRECISION(kFloat)>
    SoftmaxFp32;
REGISTER_LITE_KERNEL(softmax, kARMTrustZone, kFloat, kNCHW, SoftmaxFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARMTrustZone))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARMTrustZone))})
    .Finalize();
