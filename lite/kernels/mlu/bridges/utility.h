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

#pragma once

#include <cnml.h>
#include <cnrt.h>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/mlu/mlu_utils.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"
#include "lite/fluid/float16.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void transpose2d(float* input_data,
                 float* output_data,
                 std::vector<int> input_shape);
void transpose(float* input_data,
               float* output_data,
               std::vector<int> input_shape,
               std::vector<int> axis);

inline int scale2position(float scale) { return std::floor(-std::log2(scale)); }

void dequant(float* dst, int8_t* src, size_t size, float scale);

void dequant(float* dst,
             int8_t* src,
             size_t size_o,
             size_t size,
             size_t size_in,
             std::vector<float> scales);

template <typename T>
std::vector<T> recip(std::vector<T> x);
// Type/tensor converters for converting Paddle type/tensor to MLU type/tensor
bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

cnmlActiveFunction_t OpTypeToCNMLActType(std::string op_type);

inline const ::paddle::lite::DDimLite DimNHWC2NCHW(
    const ::paddle::lite::DDimLite& dim) {
  return ::paddle::lite::DDimLite(
      std::vector<int64_t>({dim[0], dim[3], dim[1], dim[2]}));
}

inline const ::paddle::lite::DDimLite DimNCHW2NHWC(
    const ::paddle::lite::DDimLite& dim) {
  return ::paddle::lite::DDimLite(
      std::vector<int64_t>({dim[0], dim[2], dim[3], dim[1]}));
}

inline const std::vector<int64_t> DimNHWC2NCHW(
    const std::vector<int64_t>& dim) {
  return std::vector<int64_t>({dim[0], dim[3], dim[1], dim[2]});
}

inline const std::vector<int64_t> DimNCHW2NHWC(
    const std::vector<int64_t>& dim) {
  return std::vector<int64_t>({dim[0], dim[2], dim[3], dim[1]});
}

template <paddle::lite_api::PrecisionType>
struct FPTypeTraits {};

template <>
struct FPTypeTraits<paddle::lite_api::PrecisionType::kFloat> {
  typedef float T;
};

template <>
struct FPTypeTraits<paddle::lite_api::PrecisionType::kFP16> {
  typedef paddle::lite::fluid::float16 T;
};

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
