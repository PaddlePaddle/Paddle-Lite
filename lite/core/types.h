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

#include <stack>
#include <string>
#include "lite/api/paddle_place.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace core {

/*
 * Type representations used to represent standard types.
 */
// TODO(Superjomn) unify all the type representation across the lite framework.
enum class Type {
  _unk = -1,
  // primary types
  _int32,
  _int64,
  _float32,
  _float64,
  _bool,
  _string,
  // primary list types
  _list,
  // enum type
  _enum,
  _float16,
  // number of types
  __num__,
};

template <typename T>
Type StdTypeToRepr() {
  return Type::_unk;
}
template <>
Type StdTypeToRepr<int32_t>();
template <>
Type StdTypeToRepr<int64_t>();
template <>
Type StdTypeToRepr<float>();
template <>
Type StdTypeToRepr<bool>();
template <>
Type StdTypeToRepr<std::string>();

// Factors that impact the kernel picking strategy. Multiple factors can be
// considered together by using statement like 'factor1 | factor2'
class KernelPickFactor {
 public:
  using value_type = unsigned char;
  enum class Factor : int {
    // The following factors are sorted by priority.
    TargetFirst = 1,
    PrecisionFirst = 1 << 1,
    DataLayoutFirst = 1 << 2,
    DeviceFirst = 1 << 3,
  };

  // Has any factors considered.
  bool any_factor_considered() const { return data_; }

  KernelPickFactor& ConsiderTarget();
  // Prefer a specific target, e.g. prefer CUDA kernels.
  KernelPickFactor& ConsiderPrecision();
  KernelPickFactor& ConsiderDataLayout();
  KernelPickFactor& ConsiderDevice();

  bool IsTargetConsidered() const;
  bool IsPrecisionConsidered() const;
  bool IsDataLayoutConsidered() const;
  bool IsDeviceConsidered() const;

  friend STL::ostream& operator<<(STL::ostream& os, const KernelPickFactor& k);

 private:
  unsigned char data_{};
  lite_api::TargetType target_{TARGET(kUnk)};
};

struct dim2 {
  int x{};
  int y{};

  dim2(int x, int y) : x(x), y(y) {}
};

struct dim3 {
  int x{};
  int y{};
  int z{};

  dim3(int x, int y, int z) : x(x), y(y), z(z) {}
};

using byte_t = uint8_t;

}  // namespace core
}  // namespace lite
}  // namespace paddle
