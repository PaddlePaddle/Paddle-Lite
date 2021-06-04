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

enum class DataLayout { kNCHW = 0, kNHWC };

enum class METAL_PRECISION_TYPE { FLOAT = 0, HALF = 1, INT8 = 2, INT16 = 3, INT32 = 4 };

enum class METAL_ACCESS_FLAG { CPUReadWrite = 0, CPUWriteOnly, CPUTransparent, CPUShared };

typedef uint32_t MetalUint;
typedef uint16_t MetalHalf;

struct MetalUint2 {
   public:
    uint32_t x;
    uint32_t y;

    void MaxThan1() {
        x = std::max<decltype(x)>(x, 1);
        y = std::max<decltype(y)>(y, 1);
    }
};

struct MetalUint3 {
   public:
    MetalUint x;
    MetalUint y;
    MetalUint z;

    void MaxThan1() {
        x = std::max<decltype(x)>(x, 1);
        y = std::max<decltype(y)>(y, 1);
        z = std::max<decltype(z)>(z, 1);
    }
};

struct MetalUint4 {
   public:
    MetalUint r;
    MetalUint g;
    MetalUint b;
    MetalUint a;

    void MaxThan1() {
        r = std::max<decltype(r)>(r, 1);
        g = std::max<decltype(g)>(g, 1);
        b = std::max<decltype(b)>(b, 1);
        a = std::max<decltype(a)>(a, 1);
    }
};

#define UP_DIV(x, y) (((x) + (y)-1) / (y))
#define ROUND_UP(x, y) (((x) + (y)-1) / (y) * (y))

static uint32_t SmallestLog2(uint32_t integer) {
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
