/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "operators/math/gemm/gemm_kernel.h"
#include "operators/math/gemm/pack_kernel.h"

namespace paddle_mobile {
namespace operators {
namespace math {

struct SgemmStrategy {
  typedef float Itype;
  typedef float Otype;

  typedef void (*packLhsFunc)(const int, const int, const Itype *, const int,
                              Itype *, const bool);
  typedef void (*packRhsFunc)(const int, const int, const Itype *, const int,
                              Itype *, const bool);
  typedef void (*kernelFunc)(const Itype *, const Itype *, const int, Otype *,
                             const int);
  typedef void (*WriteFunc)(const int, const int, const float alpha,
                            const Otype *, const int, const float beta, Otype *,
                            const int);

  packLhsFunc pack_lhs;
  packRhsFunc pack_rhs;
  kernelFunc kernel;
  WriteFunc write;

  static int out_width() {
#if __aarch64__
    return 16;
#else
    return 8;
#endif
  }

  static int out_height() { return 6; }

  SgemmStrategy() {
    pack_lhs = pack_lhs_6r;
#if __aarch64__
    pack_rhs = pack_rhs_16c;
    kernel = sgemm_6x16;
#else
    pack_rhs = pack_rhs_8c;
    kernel = sgemm_6x8;
#endif
    write = write_back;
  }
};

struct I8o32gemmStrategy {
  typedef int8_t Itype;
  typedef int32_t Otype;

  typedef void (*kern_type)(const Itype *, const Itype *, const int, Otype *,
                            const int);
  kern_type kernel;

  static int out_width() { return 8; }

  static int out_height() {
#if __aarch64__
    return 12;
#else
    return 6;
#endif
  }

  I8o32gemmStrategy() {}
};

struct SgemvStrategy {
  typedef float Itype;
  typedef float Otype;

  typedef void (*kernelFunc)(const bool, const int, const int, const float,
                             const Itype *, const int, const Itype *,
                             const float, Otype *);
  kernelFunc kernel;

  SgemvStrategy() { kernel = sgemv_mx1; }
};

struct I8o32gemvStrategy {
  typedef int8_t Itype;
  typedef int32_t Otype;

  typedef void (*kern_type)(const Itype *, const Itype *, const int, Otype *,
                            const int);
  kern_type kernel;

  static int out_width() { return 1; }

  static int out_height() {
#if __aarch64__
    return 12;
#else
    return 6;
#endif
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
