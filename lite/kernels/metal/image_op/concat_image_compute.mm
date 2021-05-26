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

#include "metal_params.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/concat_image_compute.h"
#include "lite/backends/metal/metal_debug.h"


namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void ConcatImageCompute<P, PTYPE>::PrepareForRun() {
  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();
  auto device = metal_context_->GetDefaultDevice();

  const auto& param = this->template Param<param_t>();
  auto output_dims = param.output->dims();

  int num = param.x.size();

  for (int i = 0; i < num; i++) {
    auto input_image = param.x[i]->template data<P, MetalImage>();
    input_buffers_.emplace_back(input_image);
  }

  output_buffer_ = param.output->template mutable_data<P, MetalImage>(output_dims);

  int axis = 4 - output_buffer_->tensor_dim_.size() + param.axis;
  auto* axis_tensor = param.axis_tensor;
  if (axis_tensor != nullptr) {
    auto* axis_tensor_data = axis_tensor->template data<int>();
    axis = axis_tensor_data[0];
  }
  if (axis < 0) {
    axis += output_buffer_->tensor_dim_.size();
  }
  auto orank = output_buffer_->tensor_dim_.size();
  assert(num <= 6);

  auto transpose = output_buffer_->transpose_;

  for (int i = 0; i < 4; i++) {
    if (transpose[i] == axis) {
      axis = i;
      break;
    }
  }

  int vdim[6] = {0, 0, 0, 0, 0, 0};
  for (int i = 0; i < num; i++) {
    vdim[i] = input_buffers_[i]->dim_[axis];
  }
  if (orank == 4) {
    if (axis == 1) {
      v_ = "y";
    } else if (axis == 2) {
      v_ = "x";
    } else {
      if ((output_buffer_->dim_[0] == 1) && (axis == 3)) {
        auto vz = true;
        for (int i = 0; i < num; i++) {
          if (vdim[i] % 4 != 0) {
            vz = false;
            break;
          }
        }
        if (vz) {
          v_ = "z";
          for (int i = 0; i < num; i++) {
            vdim[i] = vdim[i] / 4;
          }
        }
      }
    }
  } else if (orank == 3) {
    if (axis == 2) {
      v_ = "y";
    } else if (axis == 3) {
      v_ = "x";
    } else if (axis == 1) {
      auto vz = true;
      for (int i = 0; i < num; i++) {
        if (vdim[i] % 4 != 0) {
          vz = false;
          break;
        }
      }
      if (vz) {
        v_ = "z";
        for (int i = 0; i < num; i++) {
          vdim[i] = vdim[i] / 4;
        }
      }
    }
  } else {
    if (axis == 2) {
      v_ = "y";
    } else if (axis == 3) {
      bool vx = true;
      for (int i = 0; i < num; i++) {
        if (vdim[i] % 4 != 0) {
          vx = false;
          break;
        }
      }
      if (vx) {
        v_ = "x";
        for (int i = 0; i < num; i++) {
          vdim[i] = vdim[i] / 4;
        }
      }
    }
  }

  ConcatMetalParam pm{
      {(int)output_buffer_->dim_[0],
       (int)output_buffer_->dim_[1],
       (int)output_buffer_->dim_[2],
       (int)output_buffer_->dim_[3]},
      static_cast<int>(axis),
      0,
      {(int)(output_buffer_->transpose_[0]),
       (int)(output_buffer_->transpose_[1]),
       (int)(output_buffer_->transpose_[2]),
       (int)(output_buffer_->transpose_[3])},
      {(int)vdim[0], (int)vdim[1], (int)vdim[2], (int)vdim[3], (int)vdim[4], (int)vdim[5]}};

  param_buffer_ = metal_context_->CreateBuffer(*device, &pm, sizeof(pm), METAL_ACCESS_FLAG::CPUWriteOnly);

  std::string function_name = "";
  if (std::is_same<float, P>::value) {
    function_name =
        "concat_" + std::to_string(orank) + "_" + std::to_string(num) + "_" + v_ + "_float";
  } else if (std::is_same<MetalHalf, P>::value) {
    function_name =
        "concat_" + std::to_string(orank) + "_" + std::to_string(num) + "_" + v_ + "_half";
  }

  queue_ = metal_context_->GetDefaultQueue(*device);
  kernel_ = metal_context_->GetKernel(*device, function_name);
}


template <typename P, PrecisionType PTYPE>
void ConcatImageCompute<P, PTYPE>::Run() {
  auto output_width = output_buffer_->texture_width_;
  auto output_height = output_buffer_->texture_height_;
  auto output_array_length = output_buffer_->array_length_;

  auto& context = this->ctx_->template As<ContextMetal>();
  metal_context_ = (MetalContext*)context.context();

  auto encoder = std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
  MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                 static_cast<MetalUint>(output_height),
                                 static_cast<MetalUint>(output_array_length)};

  int image_index = 0;
  for (auto item : input_buffers_)
    [encoder->metal_command_encoder_ setTexture:(item->image()) atIndex:(image_index++)];
  [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(image_index++)];
  if (v_ == "normal")
    [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(image_index)];
  [encoder->metal_command_encoder_ setBuffer:(param_buffer_->buffer()) offset:(0)atIndex:(0)];

  kernel_->Execute(*encoder, global_work_size, false);
  return;
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::ConcatImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::ConcatImageCompute<MetalHalf, PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::ConcatImageCompute<float, PRECISION(kFloat)> MetalConcatFp32;
typedef paddle::lite::kernels::metal::ConcatImageCompute<MetalHalf, PRECISION(kFP16)> MetalConcatFp16;


REGISTER_LITE_KERNEL(concat,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     MetalConcatFp32,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("AxisTensor", {LiteType::GetTensorTy(TARGET(kHost),
                                                   PRECISION(kInt32),
                                                   DATALAYOUT(kNCHW))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();


REGISTER_LITE_KERNEL(concat,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     MetalConcatFp16,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("AxisTensor", {LiteType::GetTensorTy(TARGET(kHost),
                                                        PRECISION(kInt32),
                                                        DATALAYOUT(kNCHW))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();