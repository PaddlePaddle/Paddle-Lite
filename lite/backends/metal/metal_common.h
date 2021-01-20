// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_BACKENDS_METAL_METAL_COMMON_H_
#define LITE_BACKENDS_METAL_METAL_COMMON_H_

#include <algorithm>

namespace paddle {
namespace lite {

enum class METAL_PRECISION_TYPE {
  FLOAT = 0,
  HALF = 1,
  INT8 = 2,
  INT16 = 3,
  INT32 = 4
};

enum class METAL_ACCESS_FLAG { CPUReadWrite = 0, CPUWriteOnly, CPUTransparent };

typedef uint32_t metal_uint;
typedef uint16_t metal_half;

struct metal_uint2 {
 public:
  uint32_t x;
  uint32_t y;

  void max_than_1() {
    x = std::max<decltype(x)>(x, 1);
    y = std::max<decltype(y)>(y, 1);
  }
};

struct metal_uint3 {
 public:
  metal_uint x;
  metal_uint y;
  metal_uint z;

  void max_than_1() {
    x = std::max<decltype(x)>(x, 1);
    y = std::max<decltype(y)>(y, 1);
    z = std::max<decltype(z)>(z, 1);
  }
};

struct metal_uint4 {
 public:
  metal_uint r;
  metal_uint g;
  metal_uint b;
  metal_uint a;

  void max_than_1() {
    r = std::max<decltype(r)>(r, 1);
    g = std::max<decltype(g)>(g, 1);
    b = std::max<decltype(b)>(b, 1);
    a = std::max<decltype(a)>(a, 1);
  }
};

#define UP_DIV(x, y) (((x) + (y)-1) / (y))
#define ROUND_UP(x, y) (((x) + (y)-1) / (y) * (y))

static uint32_t smallest_log2(uint32_t integer) {
  if (integer == 0) return 0;
  uint32_t power = 0;
  while ((integer & 0b1) == 0) {
    integer = integer >> 1;
    power++;
  }
  return power;
}

}  // namespace lite
}  // namespace paddle
#endif  // LITE_BACKENDS_METAL_METAL_COMMON_H_
