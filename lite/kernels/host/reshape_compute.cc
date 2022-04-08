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

#include "lite/kernels/host/reshape_compute.h"
#include <vector>
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

bool CheckPositive(const DDim &dims) {
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] <= 0) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> ValidateShape(const std::vector<int64_t> &shape,
                                   const DDim &input_dims) {
  const int64_t input_size = input_dims.production();
  // only one dimension can be set to -1
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;
  int capacity = 1;
  int unk_dim_idx = -1;
  std::vector<int64_t> output_dims(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      CHECK_EQ(unk_dim_idx, -1)
          << "Only one input dimension of Attr(shape) can be unknown.";
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      CHECK_LT(i, input_dims.size())
          << "The index of dimension to copy from input shape must be less "
             "than the size of input shape.";
    } else {
      CHECK_GT(shape[i], 0) << "Each input dimension of Attr(shape) must not "
                               "be negtive except one unknown dimension.";
    }
    int64_t output_dim_i =
        shape[i] ? static_cast<int64_t>(shape[i]) : input_dims[i];
    output_dims[i] = output_dim_i;
    capacity *= output_dim_i;
  }
  if (unk_dim_idx != -1) {
    if (CheckPositive(input_dims)) {
      // input_size < 0 and is un-determinate in compile time, skip the check,
      // for example, input_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, input_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_dims[unk_dim_idx] = -input_size / capacity;
      CHECK_EQ(output_dims[unk_dim_idx] * capacity, -input_size)
          << "Invalid shape is given.";
    } else {
      output_dims[unk_dim_idx] = -1;
    }
  } else {
    CHECK_EQ(capacity, input_size) << "Invalid shape is given.";
  }
  return output_dims;
}

void ReshapeCompute::GetCurrentShape(DDim *out_shape) {
  std::vector<int64_t> final_shape{};
  auto &param = this->Param<operators::ReshapeParam>();
  const auto &shape_tensor_vct = param.shape_tensor_vct;
  auto *shape_tensor = param.shape_tensor;
  const auto &shape_vct = param.shape_vct;
  auto x_dims = param.x->dims();

  if (shape_tensor_vct.size() > 0) {
    final_shape.resize(shape_tensor_vct.size());
    for (size_t i = 0; i < shape_tensor_vct.size(); i++) {
      final_shape[i] =
          static_cast<int64_t>(shape_tensor_vct[i]->data<int>()[0]);
    }
  } else if (shape_tensor != nullptr && shape_tensor->data<int>() != nullptr) {
    auto *shape_tensor_data = shape_tensor->data<int>();
    final_shape = std::vector<int64_t>(
        shape_tensor_data, shape_tensor_data + shape_tensor->numel());
  } else if (!shape_vct.empty()) {
    final_shape.resize(shape_vct.size());
    for (size_t i = 0; i < shape_vct.size(); i++) {
      final_shape[i] = static_cast<int64_t>(shape_vct[i]);
    }
  } else {
    LOG(FATAL) << "GetCurrentShape error";
  }
  auto valid_shape = ValidateShape(final_shape, x_dims);
  out_shape->ConstructFrom(valid_shape);
}

void ReshapeCompute::ReInitWhenNeeded() {
  auto &param = this->Param<operators::ReshapeParam>();
  DDim output_shape{};
  GetCurrentShape(&output_shape);
  if (last_shape_ == output_shape) {
    return;
  }
  param.output->Resize(output_shape);
  auto out_lod = param.output->mutable_lod();
  *out_lod = param.x->lod();
  last_shape_ = output_shape;
}

void ReshapeBaseCompute::Run() {
  auto &param = this->Param<operators::ReshapeParam>();
  auto x = param.x;
  auto output = param.output;
  auto output_dims = output->dims();
  auto output_lod = output->lod();
  // printf("param.inplace: %d \n", param.inplace);
  if (param.inplace) {
    output->ShareDataWith(*x);
  } else {
    auto input_ptype = x->precision();
    output->set_precision(input_ptype);
    size_t data_size = 0;
    switch (input_ptype) {
      case PrecisionType::kFloat:
      case PrecisionType::kInt32:
      case PrecisionType::kBool:
        data_size = 4;
        break;
      case PrecisionType::kUInt8:
        data_size = 1;
        break;
      case PrecisionType::kInt16:
      case PrecisionType::kFP16:
        data_size = 2;
        break;
      case PrecisionType::kUnk:
      case PrecisionType::kAny:
      case PrecisionType::kInt64:
        data_size = 8;
        break;
      default:
        LOG(FATAL) << "not support input type";
    }
    data_size = data_size * x->dims().production();
    output->mutable_data(data_size);
    output->CopyDataFrom(*x);
  }
  // printf("output_dims: %d\n", output_dims.size());
  output->Resize(output_dims);
  // printf("resize end\n");
  output->set_lod(output_lod);
  // printf("lod end\n");
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using ReshapeReshape = paddle::lite::kernels::host::ReshapeCompute;
REGISTER_LITE_KERNEL(reshape, kHost, kAny, kAny, ReshapeReshape, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2, kHost, kAny, kAny, ReshapeReshape, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

using ReshapeFlatten = paddle::lite::kernels::host::FlattenCompute;
REGISTER_LITE_KERNEL(flatten, kHost, kAny, kAny, ReshapeFlatten, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten2, kHost, kAny, kAny, ReshapeFlatten, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
