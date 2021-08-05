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

#include "lite/kernels/host/gather_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename IndexType, typename DataType>
void GatherFunc(const operators::GatherParam& param) {
  auto src_dims = param.X->dims();
  auto index_size = param.Index->dims()[0];
  auto* p_src = param.X->data<DataType>();
  const IndexType* p_index = param.Index->data<IndexType>();
  auto* p_output = param.Out->mutable_data<DataType>();

  int slice_size = 1;
  for (size_t i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  for (int i = 0; i < index_size; ++i) {
    IndexType index_ = p_index[i];
    memcpy(p_output + i * slice_size,
           p_src + index_ * slice_size,
           slice_size * sizeof(DataType));
  }
}

template <typename IndexType, typename AxisType, typename DataType>
void GatherV2Func(const operators::GatherParam& param, const int axis) {
  auto* index_data = param.Index->data<IndexType>();
  auto* input_data = param.X->data<DataType>();
  auto* out_data = param.Out->mutable_data<DataType>();

  int index_size = param.Index->numel();
  int input_size = param.X->numel();
  if (input_size == 0) return;
  auto input_dim = param.X->dims();
  const int axis_index = axis;
  int input_index_dim_size = input_dim[axis_index];
  for (int i = 0; i < index_size; i++) {
    LOG(INFO) << lite::string_format("index_data[%d]: %lld", i, index_data[i]);
    CHECK_LT(index_data[i], input_index_dim_size)
        << "The element of Index must be less than the size of "
        << "dim size of axis dim";
  }

  int inner_dim_size = 1;
  int outer_dim_size = 1;
  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
  }
  for (size_t i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
  }

  int out_index = 0;
  for (int i = 0; i < inner_dim_size; i++) {
    for (int j = 0; j < index_size; j++) {
      for (int k = 0; k < outer_dim_size; k++) {
        int index = k + index_data[j] * outer_dim_size +
                    (i * input_size / inner_dim_size);
        out_data[out_index] = input_data[index];
        out_index++;
      }
    }
  }
}

template <typename IndexType, typename AxisType>
void GatherCompute<IndexType, AxisType>::Run() {
  auto& param = this->template Param<operators::GatherParam>();

  int axis = param.axis;
  // get axis from tensor
  if (param.Axis != nullptr) {
    LOG(INFO) << "param.Axis is not nullptr";
    const Tensor* axis_tensor = param.Axis;
    const auto& axis_type = axis_tensor->precision();
    if (axis_type == PRECISION(kInt32)) {
      axis = static_cast<int>(axis_tensor->data<int32_t>()[0]);
    } else if (axis_type == PRECISION(kInt64)) {
      axis = static_cast<int>(axis_tensor->data<int64_t>()[0]);
    } else {
      LOG(FATAL) << "unsupport data type of Axis tensor: "
                 << lite_api::PrecisionToStr(axis_type);
    }
  }

  LOG(INFO) << "axis: " << axis;
  LOG(INFO) << "in dims: " << param.X->dims();
  LOG(INFO) << "index dims: " << param.Index->dims();
  LOG(INFO) << "out dims: " << param.Out->dims();

  const auto& data_type = param.X->precision();
  if (axis != 0) {
    switch (data_type) {
      case PRECISION(kFloat):
        GatherV2Func<IndexType, AxisType, float>(param, axis);
        break;
#ifdef ENABLE_ARM_FP16
      case PRECISION(kFP16):
        GatherV2Func<IndexType, AxisType, lite_api::float16_t>(param, axis);
        break;
#endif
      case PRECISION(kInt8):
        GatherV2Func<IndexType, AxisType, int8_t>(param, axis);
        break;
      case PRECISION(kInt16):
        GatherV2Func<IndexType, AxisType, int16_t>(param, axis);
        break;
      case PRECISION(kInt32):
        GatherV2Func<IndexType, AxisType, int32_t>(param, axis);
        break;
      case PRECISION(kInt64):
        GatherV2Func<IndexType, AxisType, int64_t>(param, axis);
        break;
      default:
        LOG(FATAL) << "unsupport data type: "
                   << lite_api::PrecisionToStr(data_type);
    }
    return;
  }

  if (param.X->numel() == 0) return;
  switch (data_type) {
    case PRECISION(kFloat):
      GatherFunc<IndexType, float>(param);
      break;
#ifdef ENABLE_ARM_FP16
    case PRECISION(kFP16):
      GatherFunc<IndexType, lite_api::float16_t>(param);
      break;
#endif
    case PRECISION(kInt8):
      GatherFunc<IndexType, int8_t>(param);
      break;
    case PRECISION(kInt16):
      GatherFunc<IndexType, int16_t>(param);
      break;
    case PRECISION(kInt32):
      GatherFunc<IndexType, int32_t>(param);
      break;
    case PRECISION(kInt64):
      GatherFunc<IndexType, int64_t>(param);
      break;
    default:
      LOG(FATAL) << "unsupport data type: "
                 << lite_api::PrecisionToStr(data_type);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(gather, kHost, kAny, kAny, GatherInt32Int32, int32int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(gather, kHost, kAny, kAny, GatherInt64Int64, int64int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(gather, kHost, kAny, kAny, GatherInt64Int32, int64int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(gather, kHost, kAny, kAny, GatherInt32Int64, int32int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
