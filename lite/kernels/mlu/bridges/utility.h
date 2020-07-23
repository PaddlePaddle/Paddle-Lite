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
void transpose(dtype* input_data,
               dtype* output_data,
               std::vector<int> input_shape,
               std::vector<int> axis) {
  int old_index = -1;
  int new_index = -1;
  std::vector<int> shape;
  std::vector<int> expand_axis;
  if (input_shape.size() < 5u) {
    for (size_t i = 0; i < 5 - input_shape.size(); i++) {
      shape.push_back(1);
      expand_axis.push_back(i);
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      shape.push_back(input_shape[i]);
      expand_axis.push_back(axis[i] + 5 - input_shape.size());
    }
  } else {
    shape = input_shape;
    expand_axis = axis;
  }
  int dim[5] = {0};
  for (dim[0] = 0; dim[0] < shape[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < shape[1]; dim[1]++) {
      for (dim[2] = 0; dim[2] < shape[2]; dim[2]++) {
        for (dim[3] = 0; dim[3] < shape[3]; dim[3]++) {
          for (dim[4] = 0; dim[4] < shape[4]; dim[4]++) {
            old_index = dim[0] * shape[1] * shape[2] * shape[3] * shape[4] +
                        dim[1] * shape[2] * shape[3] * shape[4] +
                        dim[2] * shape[3] * shape[4] + dim[3] * shape[4] +
                        dim[4];
            new_index = dim[expand_axis[0]] * shape[expand_axis[1]] *
                            shape[expand_axis[2]] * shape[expand_axis[3]] *
                            shape[expand_axis[4]] +
                        dim[expand_axis[1]] * shape[expand_axis[2]] *
                            shape[expand_axis[3]] * shape[expand_axis[4]] +
                        dim[expand_axis[2]] * shape[expand_axis[3]] *
                            shape[expand_axis[4]] +
                        dim[expand_axis[3]] * shape[expand_axis[4]] +
                        dim[expand_axis[4]];
            output_data[new_index] = input_data[old_index];
          }
        }
      }
    }
  }
}

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

template <typename data_type>
inline const std::vector<data_type> DimNHWC2NCHW(
    const std::vector<data_type>& dim) {
  switch (dim.size()) {
    case 1:
      return dim;
    case 2:
      return dim;
    case 3:
      return std::vector<data_type>({dim[0], dim[2], dim[1]});
    case 4:
      return std::vector<data_type>({dim[0], dim[3], dim[1], dim[2]});
    case 5:
      return std::vector<data_type>({dim[0], dim[4], dim[1], dim[2], dim[3]});
    default:
      CHECK(0) << "unsupport dimension";
  }
}

template <typename data_type>
inline const std::vector<data_type> DimNCHW2NHWC(
    const std::vector<data_type>& dim) {
  switch (dim.size()) {
    case 1:
      return dim;
    case 2:
      return dim;
    case 3:
      return std::vector<data_type>({dim[0], dim[2], dim[1]});
    case 4:
      return std::vector<data_type>({dim[0], dim[2], dim[3], dim[1]});
    case 5:
      return std::vector<data_type>({dim[0], dim[2], dim[3], dim[4], dim[1]});
    default:
      CHECK(0) << "unsupport dimension";
  }
}

template <typename data_type>
inline std::vector<data_type> GetAxisNHWC2NCHW(size_t n_dims) {
  std::vector<data_type> nhwc2nchw_axis(n_dims);
  nhwc2nchw_axis[0] = 0;
  if (n_dims > 1) nhwc2nchw_axis[1] = n_dims - 1;
  for (size_t i = 2; i < n_dims; ++i) {
    nhwc2nchw_axis[i] = i - 1;
  }
  return nhwc2nchw_axis;
}

template <typename data_type>
inline std::vector<data_type> GetAxisNCHW2NHWC(size_t n_dims) {
  std::vector<data_type> nchw2nhwc_axis(n_dims);
  nchw2nhwc_axis[0] = 0;
  for (size_t i = 1; i < n_dims - 1; ++i) {
    nchw2nhwc_axis[i] = i + 1;
  }
  if (n_dims > 1) nchw2nhwc_axis[n_dims - 1] = 1;
  return nchw2nhwc_axis;
}

template <paddle::lite_api::PrecisionType>
struct MLUTypeTraits {
  /* using type = void; */
  /* static constexpr cnmlDataType_t cnml_type = CNML_DATA_INVALID; */
};

template <>
struct MLUTypeTraits<paddle::lite_api::PrecisionType::kFloat> {
  using type = float;
  static constexpr cnmlDataType_t cnml_type = CNML_DATA_FLOAT32;
};

template <>
struct MLUTypeTraits<paddle::lite_api::PrecisionType::kFP16> {
  using type = paddle::lite::fluid::float16;
  static constexpr cnmlDataType_t cnml_type = CNML_DATA_FLOAT16;
};

template <>
struct MLUTypeTraits<paddle::lite_api::PrecisionType::kInt8> {
  using type = int8_t;
  static constexpr cnmlDataType_t cnml_type = CNML_DATA_INT8;
};

template <>
struct MLUTypeTraits<paddle::lite_api::PrecisionType::kInt32> {
  using type = int32_t;
  static constexpr cnmlDataType_t cnml_type = CNML_DATA_INT32;
};

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
