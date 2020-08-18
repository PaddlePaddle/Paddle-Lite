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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/huawei_ascend_npu/utils.h"
#include "lite/core/op_lite.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

#define INPUT_UPDATE(...) TENSOR_INPUT_UPDATE(__VA_ARGS__, ge::FORMAT_NCHW)
#define OUTPUT_UPDATE(...) TENSOR_OUTPUT_UPDATE(__VA_ARGS__, ge::FORMAT_NCHW)
#define DYNAMIC_INPUT_UPDATE(...) \
  TENSOR_DYNAMIC_INPUT_UPDATE(__VA_ARGS__, ge::FORMAT_NCHW)
#define DYNAMIC_OUTPUT_UPDATE(...) \
  TENSOR_DYNAMIC_OUTPUT_UPDATE(__VA_ARGS__, ge::FORMAT_NCHW)

#define TENSOR_INPUT_UPDATE(op, attr, node, format)              \
  ge::TensorDesc _##op##_input_desc_##attr(                      \
      ge::Shape(), format, CvtPrecisionType(node->precision())); \
  _##op##_input_desc_##attr.SetName(node->name());               \
  op->update_input_desc_##attr(_##op##_input_desc_##attr);
#define TENSOR_OUTPUT_UPDATE(op, attr, node, format)             \
  ge::TensorDesc _##op##_output_desc_##attr(                     \
      ge::Shape(), format, CvtPrecisionType(node->precision())); \
  _##op##_output_desc_##attr.SetName(node->name());              \
  op->update_output_desc_##attr(_##op##_output_desc_##attr);
#define TENSOR_DYNAMIC_INPUT_UPDATE(op, attr, idx, node, format) \
  ge::TensorDesc _##op##_input_desc_##attr##_##idx(              \
      ge::Shape(), format, CvtPrecisionType(node->precision())); \
  _##op##_input_desc_##attr##_##idx.SetName(node->name());       \
  op->update_dynamic_input_desc_##attr(idx, _##op##_input_desc_##attr##_##idx);
#define TENSOR_DYNAMIC_OUTPUT_UPDATE(op, attr, idx, node, format) \
  ge::TensorDesc _##op##_output_desc_##attr##_##idx(              \
      ge::Shape(), format, CvtPrecisionType(node->precision()));  \
  _##op##_output_desc_##attr##_##idx.SetName(node->name());       \
  op->update_dynamic_output_desc_##attr(idx,                      \
                                        _##op##_output_desc_##attr##_##idx);

// Type/tensor converters for converting Paddle type/tensor to HiAI type/tensor
bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

ge::DataType CvtPrecisionType(PrecisionType itype);

ge::Format CvtDataLayoutType(DataLayoutType itype);

// Padding the shape to 4-dimensions(NCHW) for Huawei Ascend NPU
std::vector<int64_t> CvtShape(const std::vector<int64_t>& in_shape);

std::vector<int64_t> CvtShape(const DDim& in_dims);

ge::Tensor CvtTensor(const Tensor& in_tensor,
                     std::vector<int64_t> out_shape = {},
                     DataLayoutType in_layout = DATALAYOUT(kNCHW));

int CvtActMode(std::string act_type);

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
