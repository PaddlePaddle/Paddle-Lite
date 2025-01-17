//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "lite/backends/loongarch/xxl.h"
#include "lite/backends/loongarch/cpu_info.h"
#include "lite/backends/loongarch/math/lsx/lsx_mathfuns.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {
/* __m256 is ugly to write */
typedef __m256 v8sf;   // vector of 8 float
typedef __m256i v8si;  // vector of 8 int
v8sf log256_ps(v8sf x);
v8sf exp256_ps(v8sf x);
v8sf pow256_ps(v8sf x, v8sf y);
v8sf sin256_ps(v8sf x);
v8sf cos256_ps(v8sf x);
void sincos256_ps(v8sf x, v8sf *s, v8sf *c);

#define lasx_loadu_epi8(ptr) \
  lasx_loadu_m256i(reinterpret_cast<__m256i const *>(ptr))
#define lasx_storeu_epi8(ptr) \
  lasx_storeu_m256i(reinterpret_cast<__m256i const *>(ptr))

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
