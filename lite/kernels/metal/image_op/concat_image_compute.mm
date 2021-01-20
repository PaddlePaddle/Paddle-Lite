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

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void concat_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();

  int num = param.x.size();

  for (int i = 0; i < num; i++) {
    auto input_image = param.x[i]->data<float, metal_image>();
    input_buffers_.emplace_back(input_image);
  }
  output_buffer_ = param.output->mutable_data<float, metal_image>(output_dims);

  int axis = param.axis;
  auto* axis_tensor = param.axis_tensor;
  if (axis_tensor != nullptr) {
    auto* axis_tensor_data = axis_tensor->data<int>();
    axis = axis_tensor_data[0];
  }
  if (axis < 0) {
    axis += output_buffer_->tensorDim_.size();
  }
  auto orank = output_buffer_->tensorDim_.size();
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
      v = "y";
    } else if (axis == 2) {
      v = "x";
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
          v = "z";
          for (int i = 0; i < num; i++) {
            vdim[i] = vdim[i] / 4;
          }
        }
      }
    }
  } else if (orank == 3) {
    if (axis == 2) {
      v = "y";
    } else if (axis == 3) {
      v = "x";
    } else if (axis == 1) {
      auto vz = true;
      for (int i = 0; i < num; i++) {
        if (vdim[i] % 4 != 0) {
          vz = false;
          break;
        }
      }
      if (vz) {
        v = "z";
        for (int i = 0; i < num; i++) {
          vdim[i] = vdim[i] / 4;
        }
      }
    }
  } else {
    if (axis == 2) {
      v = "y";
    } else if (axis == 3) {
      bool vx = true;
      for (int i = 0; i < num; i++) {
        if (vdim[i] % 4 != 0) {
          vx = false;
          break;
        }
      }
      if (vx) {
        v = "x";
        for (int i = 0; i < num; i++) {
          vdim[i] = vdim[i] / 4;
        }
      }
    }
  }

  ConcatMetalParam pm{
      {(int)param.output->dims()[0],
       (int)param.output->dims()[1],
       (int)param.output->dims()[2],
       (int)param.output->dims()[3]},
      static_cast<int>(axis),
      0,
      {(int)(output_buffer_->transpose_[0]),
       (int)(output_buffer_->transpose_[1]),
       (int)(output_buffer_->transpose_[2]),
       (int)(output_buffer_->transpose_[3])},
      {(int)vdim[0], (int)vdim[1], (int)vdim[2], (int)vdim[3], (int)vdim[4], (int)vdim[5]}};

  param_buffer_ = mtl_ctx->create_buffer(*device, &pm, sizeof(pm), METAL_ACCESS_FLAG::CPUWriteOnly);

  string function_name =
      "concat_" + std::to_string(orank) + "_" + std::to_string(num) + "_" + v + "_float";
  kernel_ = mtl_ctx->get_kernel(*device, function_name);
}

void concat_image_compute::Run() {
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    // TODO: check the vector input failed reason
    std::vector<metal_kernel_arg> args;
    for (auto item : input_buffers_) args.emplace_back(item);
    args.emplace_back(output_buffer_);
    args.emplace_back(param_buffer_);

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

void concat_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();

  int num = param.x.size();

  for (int i = 0; i < num; i++) {
    auto input_image = param.x[i]->data<metal_half, metal_image>();
    input_buffers_.emplace_back(input_image);
  }
  output_buffer_ = param.output->mutable_data<metal_half, metal_image>(output_dims);

  int axis = param.axis;
  auto* axis_tensor = param.axis_tensor;
  if (axis_tensor != nullptr) {
    auto* axis_tensor_data = axis_tensor->data<int>();
    axis = axis_tensor_data[0];
  }
  if (axis < 0) {
    axis += output_buffer_->tensorDim_.size();
  }
  auto orank = output_buffer_->tensorDim_.size();
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
      v = "y";
    } else if (axis == 2) {
      v = "x";
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
          v = "z";
          for (int i = 0; i < num; i++) {
            vdim[i] = vdim[i] / 4;
          }
        }
      }
    }
  } else if (orank == 3) {
    if (axis == 2) {
      v = "y";
    } else if (axis == 3) {
      v = "x";
    } else if (axis == 1) {
      auto vz = true;
      for (int i = 0; i < num; i++) {
        if (vdim[i] % 4 != 0) {
          vz = false;
          break;
        }
      }
      if (vz) {
        v = "z";
        for (int i = 0; i < num; i++) {
          vdim[i] = vdim[i] / 4;
        }
      }
    }
  } else {
    if (axis == 2) {
      v = "y";
    } else if (axis == 3) {
      bool vx = true;
      for (int i = 0; i < num; i++) {
        if (vdim[i] % 4 != 0) {
          vx = false;
          break;
        }
      }
      if (vx) {
        v = "x";
        for (int i = 0; i < num; i++) {
          vdim[i] = vdim[i] / 4;
        }
      }
    }
  }

  ConcatMetalParam pm{
      {(int)param.output->dims()[0],
       (int)param.output->dims()[1],
       (int)param.output->dims()[2],
       (int)param.output->dims()[3]},
      static_cast<int>(axis),
      0,
      {(int)(output_buffer_->transpose_[0]),
       (int)(output_buffer_->transpose_[1]),
       (int)(output_buffer_->transpose_[2]),
       (int)(output_buffer_->transpose_[3])},
      {(int)vdim[0], (int)vdim[1], (int)vdim[2], (int)vdim[3], (int)vdim[4], (int)vdim[5]}};

  param_buffer_ = mtl_ctx->create_buffer(*device, &pm, sizeof(pm), METAL_ACCESS_FLAG::CPUWriteOnly);

  string function_name =
      "concat_" + std::to_string(orank) + "_" + std::to_string(num) + "_" + v + "_half";
  kernel_ = mtl_ctx->get_kernel(*device, function_name);
}

void concat_image_compute_half::Run() {
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    // TODO: check the vector input failed reason
    std::vector<metal_kernel_arg> args;
    for (auto item : input_buffers_) args.emplace_back(item);
    args.emplace_back(output_buffer_);
    args.emplace_back(param_buffer_);

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(concat,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::concat_image_compute,
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
                     paddle::lite::kernels::metal::concat_image_compute_half,
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