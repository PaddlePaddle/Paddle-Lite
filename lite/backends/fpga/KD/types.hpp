/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

namespace paddle {
namespace zynqmp {

enum DataType : int {
  FP32 = 0,
  FP16 = 1,
  INT8 = 2,
  INT16 = 3,
  INT32 = 4,
  INT64 = 5,
};

template <typename T>
struct TypeResolver {
  DataType operator()() { return FP32; }
};

template <>
struct TypeResolver<float> {
  DataType operator()() { return FP32; }
};

template <>
struct TypeResolver<float16> {
  DataType operator()() { return FP16; }
};

template <>
struct TypeResolver<int8_t> {
  DataType operator()() { return INT8; }
};

template <>
struct TypeResolver<int> {
  DataType operator()() { return INT32; }
};

template <>
struct TypeResolver<int64_t> {
  DataType operator()() { return INT64; }
};

inline int CellSize(DataType type) {
  switch (type) {
    case FP32:
      return sizeof(float);
    case FP16:
      return sizeof(float16);
    case INT32:
      return sizeof(int32_t);
    case INT8:
      return sizeof(int8_t);
    case INT16:
      return sizeof(int16_t);
    case INT64:
      return sizeof(int64_t);
    default:
      exit(-1);
      return 0;
  }
  return 0;
}

}  // namespace zynqmp
}  // namespace paddle
