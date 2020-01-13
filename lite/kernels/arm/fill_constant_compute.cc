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

#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class FillConstantCompute : public KernelLite<TARGET(kARM), PRECISION(kAny)> {
 public:
  using param_t = operators::FillConstantParam;

  inline DDimLite GetShape(const param_t& param) {
    // 1. shape is a Tensor
    if (param.shape_tensor != nullptr) {
      auto* shape_tensor = param.shape_tensor;
      auto* shape_data = shape_tensor->data<int>();
      auto vec_shape =
          std::vector<int64_t>(shape_data, shape_data + shape_tensor->numel());
      return DDimLite(vec_shape);
    }

    // 2. shape is a list/tuple containing Tensor
    auto shape_tensor_list = param.shape_tensor_list;
    if (shape_tensor_list.size() > 0) {
      std::vector<int64_t> vec_shape;
      for (size_t i = 0; i < shape_tensor_list.size(); ++i) {
        auto tensor = shape_tensor_list[i];
        vec_shape.push_back(*tensor->data<int>());
      }
      return DDimLite(vec_shape);
    }

    // 3. shape is a list/tuple without containing Tensor
    auto vec_shape = param.shape;
    return DDimLite(vec_shape);
  }

  void PrepareForRun() override {
    auto& param = *param_.get_mutable<param_t>();
    auto outdims = GetShape(param);
    param.Out->Resize(outdims);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<ARMContext>();

    if (param.dtype == static_cast<int32_t>(lite::core::FluidType::INT8)) {
      auto data = param.Out->template mutable_data<int8_t>();
      for (int i = 0; i < param.Out->numel(); i++) {
        data[i] = param.value;
      }
    } else {
      auto data = param.Out->template mutable_data<float>();
      for (int i = 0; i < param.Out->numel(); i++) {
        data[i] = param.value;
      }
    }
  }

  virtual ~FillConstantCompute() = default;
};

class FillConstantBatchLikeCompute
    : public KernelLite<TARGET(kARM), PRECISION(kAny)> {
 public:
  using param_t = operators::FillConstantBatchLikeParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<ARMContext>();

    if (param.input->lod().size() && param.input_dim_idx == 0) {
      auto odims = param.out->dims();
      odims[param.output_dim_idx] = param.input->lod().back().size() - 1;
      param.out->Resize(odims);
    }

    if (param.dtype == static_cast<int32_t>(lite::core::FluidType::FP32)) {
      auto data = param.out->template mutable_data<float>();
      for (int i = 0; i < param.out->numel(); i++) {
        data[i] = param.value;
      }
    } else if (param.dtype ==
               static_cast<int32_t>(lite::core::FluidType::INT32)) {
      auto data = param.out->template mutable_data<int32_t>();
      for (int i = 0; i < param.out->numel(); i++) {
        data[i] = param.value;
      }
    } else if (param.dtype ==
               static_cast<int32_t>(lite::core::FluidType::INT8)) {
      auto data = param.out->template mutable_data<int8_t>();
      for (int i = 0; i < param.out->numel(); i++) {
        data[i] = param.value;
      }
    } else {
      LOG(FATAL) << "not supported dtype " << param.dtype;
    }
  }

  virtual ~FillConstantBatchLikeCompute() = default;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(fill_constant,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::FillConstantCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("ShapeTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
REGISTER_LITE_KERNEL(fill_constant_batch_size_like,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::FillConstantBatchLikeCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
