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

#include "lite/kernels/arm/concat_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

std::vector<size_t> stride_numel(const DDim& ddim) {
  std::vector<size_t> strides(ddim.size());
  strides[ddim.size() - 1] = ddim[ddim.size() - 1];
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i];
  }
  return strides;
}

template <typename T>
void ConcatFunc(const std::vector<lite::Tensor*> inputs,
                int axis,
                lite::Tensor* out) {
  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && inputs.size() < 10) {
    size_t output_offset = 0;
    for (auto* in : inputs) {
      auto in_stride = stride_numel(in->dims());
      auto out_stride = stride_numel(out->dims());
      void* dst = out->mutable_data<T>() + output_offset;
      const void* src = in->data<T>();
      // src and dst tensor should have the same dims size.
      CHECK(in_stride.size() == out_stride.size());
      std::memcpy(dst, src, sizeof(T) * in_stride[0]);
      output_offset += in_stride[0];
    }
  } else {
    lite::arm::math::concat_func<T>(inputs, axis, out);
  }
}

void ConcatCompute::Run() {
  auto& param = Param<operators::ConcatParam>();
  std::vector<lite::Tensor*> inputs = param.x;
  CHECK_GE(inputs.size(), 1);
  auto* out = param.output;
  int axis = param.axis;
  auto* axis_tensor = param.axis_tensor;
  if (axis_tensor != nullptr) {
    auto* axis_tensor_data = axis_tensor->data<int>();
    axis = axis_tensor_data[0];
  }
  if (axis < 0) {
    axis += static_cast<int>(inputs[0]->dims().size());
  }

  lite_api::PrecisionType type = PRECISION(kUnk);
  for (auto* tensor : inputs) {
    if (tensor->IsInitialized() && tensor->numel() > 0) {
      if (type == PRECISION(kUnk)) {
        type = tensor->precision();
      } else {
        VLOG(4) << "type: " << PrecisionRepr(type)
                << ", tensor: " << PrecisionRepr(tensor->precision());
#ifdef ENABLE_ARM_FP16
        if (type != tensor->precision()) {
          if (tensor->precision() == PrecisionType::kFP16 &&
              type == PrecisionType::kFloat) {
            // if last tensor's precision is fp32 && this tensor's precision is
            // fp16, do fp16->fp32 for this tensor
            Tensor tmp;
            auto shape = tensor->dims();
            tmp.Resize(shape);
            tmp.set_precision(PrecisionType::kFloat);
            auto dout = tmp.mutable_data<float>();
            auto din = tensor->data<float16_t>();
            auto size = tmp.numel();
            lite::arm::math::fp16::fp16_to_fp32(din, dout, tensor->numel());
            tensor->CopyDataFrom(tmp);
            continue;
          } else if (tensor->precision() == PrecisionType::kFloat &&
                     type == PrecisionType::kFP16) {
            // if last tensor's precision is fp16 && this tensor's precision is
            // fp32, do fp32->fp16 for this tensor
            Tensor tmp;
            auto shape = tensor->dims();
            tmp.Resize(shape);
            tmp.set_precision(PrecisionType::kFP16);
            auto dout = tmp.mutable_data<float16_t>();
            auto din = tensor->data<float>();
            auto size = tmp.numel();
            lite::arm::math::fp16::fp32_to_fp16(din, dout, tensor->numel());
            tensor->CopyDataFrom(tmp);
            continue;
          }
        }
#endif
        CHECK(type == tensor->precision()) << "The precision of "
                                           << "concat inputs should be same.";
      }
    }
  }

  switch (type) {
    case PRECISION(kFloat):
      ConcatFunc<float>(inputs, axis, out);
      break;
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      ConcatFunc<__fp16>(inputs, axis, out);
      break;
#endif
    case PRECISION(kInt32):
      ConcatFunc<int32_t>(inputs, axis, out);
      break;
    case PRECISION(kInt64):
      ConcatFunc<int64_t>(inputs, axis, out);
      break;
    case PRECISION(kBool):
      ConcatFunc<bool>(inputs, axis, out);
      break;
    default:
      LOG(FATAL) << "Concat does not implement for the "
                 << "input type:"
                 << static_cast<int>(inputs.front()->precision());
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    concat, kARM, kAny, kNCHW, paddle::lite::kernels::arm::ConcatCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
