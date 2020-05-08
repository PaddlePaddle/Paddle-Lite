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
template <typename dtype>
void transpose(dtype input_data,
               dtype output_data,
               std::vector<int> input_shape,
               std::vector<int> axis);

template <typename dtype>
void transpose(dtype input_data,
               dtype output_data,
               std::vector<int> input_shape,
               std::vector<int> axis) {
  int old_index = -1;
  int new_index = -1;
  int dim[4] = {0};
  std::vector<int> shape = input_shape;
  for (dim[0] = 0; dim[0] < input_shape[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < input_shape[1]; dim[1]++) {
      for (dim[2] = 0; dim[2] < input_shape[2]; dim[2]++) {
        for (dim[3] = 0; dim[3] < input_shape[3]; dim[3]++) {
          old_index = dim[0] * shape[1] * shape[2] * shape[3] +
                      dim[1] * shape[2] * shape[3] + dim[2] * shape[3] + dim[3];
          new_index =
              dim[axis[0]] * shape[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[2]] * shape[axis[3]] + dim[axis[3]];
          output_data[new_index] = input_data[old_index];
        }
      }
    }
  }
}

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

inline const std::vector<DDimLite::value_type> DimNHWC2NCHW(
    const std::vector<DDimLite::value_type>& dim) {
  switch (dim.size()) {
    case 1:
      return dim;
    case 2:
      return dim;
    case 3:
      return std::vector<DDimLite::value_type>({dim[0], dim[2], dim[1]});
    case 4:
      return std::vector<DDimLite::value_type>(
          {dim[0], dim[3], dim[1], dim[2]});
    case 5:
      return std::vector<DDimLite::value_type>(
          {dim[0], dim[4], dim[1], dim[2], dim[3]});
    default:
      CHECK(0) << "unsupport dimension";
  }
}

inline const std::vector<DDimLite::value_type> DimNCHW2NHWC(
    const std::vector<DDimLite::value_type>& dim) {
  switch (dim.size()) {
    case 1:
      return dim;
    case 2:
      return dim;
    case 3:
      return std::vector<DDimLite::value_type>({dim[0], dim[2], dim[1]});
    case 4:
      return std::vector<DDimLite::value_type>(
          {dim[0], dim[2], dim[3], dim[1]});
    case 5:
      return std::vector<DDimLite::value_type>(
          {dim[0], dim[2], dim[3], dim[4], dim[1]});
    default:
      CHECK(0) << "unsupport dimension";
  }
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
