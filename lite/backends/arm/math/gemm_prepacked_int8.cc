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

#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include <arm_neon.h>
#include "lite/core/parallel_defines.h"
#ifdef __aarch64__
#include "lite/backends/arm/math/dotprod/gemm_sdot.h"
#else
#include "lite/backends/arm/math/dotprod/gemm_vsdot.h"
#endif
namespace paddle {
namespace lite {
namespace arm {
namespace math {

void prepackA_m4k2x2_int8(int8_t* out,
                          const int8_t* in,
                          int ldin,
                          int m0,
                          int mmax,
                          int k0,
                          int kmax);

void prepackA_m4k2x2_trans_int8(int8_t* out,
                                const int8_t* in,
                                int ldin,
                                int m0,
                                int mmax,
                                int k0,
                                int kmax);

void packb_int8(int8_t* out,
                const int8_t* in,
                int ldin,
                int k0,
                int kmax,
                int n0,
                int nmax,
                const int8_t* zerobuf);

void packb_trans_int8(int8_t* out,
                      const int8_t* in,
                      int ldin,
                      int k0,
                      int kmax,
                      int n0,
                      int nmax,
                      const int8_t* zerobuf);

#ifdef WITH_ARM_DOTPROD
void prepackA_m8k4_int8(int8_t* out,
                        const int8_t* in,
                        int ldin,
                        int m0,
                        int mmax,
                        int k0,
                        int kmax);

void prepackA_m8k4_trans_int8(int8_t* out,
                              const int8_t* in,
                              int ldin,
                              int m0,
                              int mmax,
                              int k0,
                              int kmax);

void packb_sdot_int8_n12_n8_n4_trans(int8_t* out,
                                     const int8_t* in,
                                     int ldin,
                                     int k0,
                                     int kmax,
                                     int n0,
                                     int nmax);

void packb_sdot_int8_n12_n8_n4(int8_t* out,
                               const int8_t* in,
                               int ldin,
                               int k0,
                               int kmax,
                               int n0,
                               int nmax);

void packb_sdot_int8(int8_t* out,
                     const int8_t* in,
                     int ldin,
                     int k0,
                     int kmax,
                     int n0,
                     int nmax);

void packb_sdot_trans_int8(int8_t* out,
                           const int8_t* in,
                           int ldin,
                           int k0,
                           int kmax,
                           int n0,
                           int nmax);
void prepackA_m6k4_int8(int8_t* out,
                        const int8_t* in,
                        int ldin,
                        int m0,
                        int mmax,
                        int k0,
                        int kmax);
void prepackA_m6k4_trans_int8(int8_t* out,
                              const int8_t* in,
                              int ldin,
                              int m0,
                              int mmax,
                              int k0,
                              int kmax);
#endif

void prepackA_int8(void* out,
                   const void* in,
                   int ldin,
                   int m0,
                   int mmax,
                   int k0,
                   int kmax,
                   bool is_trans,
                   ARMContext* ctx) {
#ifdef __aarch64__
  if (ctx->has_dot()) {
#ifdef WITH_ARM_DOTPROD
    if (is_trans) {
      prepackA_m8k4_trans_int8(static_cast<int8_t*>(out),
                               static_cast<const int8_t*>(in),
                               ldin,
                               m0,
                               mmax,
                               k0,
                               kmax);
    } else {
      prepackA_m8k4_int8(static_cast<int8_t*>(out),
                         static_cast<const int8_t*>(in),
                         ldin,
                         m0,
                         mmax,
                         k0,
                         kmax);
    }
#endif
  } else {
    if (is_trans) {
      prepackA_m4k2x2_trans_int8(static_cast<int8_t*>(out),
                                 static_cast<const int8_t*>(in),
                                 ldin,
                                 m0,
                                 mmax,
                                 k0,
                                 kmax);
    } else {
      prepackA_m4k2x2_int8(static_cast<int8_t*>(out),
                           static_cast<const int8_t*>(in),
                           ldin,
                           m0,
                           mmax,
                           k0,
                           kmax);
    }
  }

#else
  if (ctx->has_dot()) {
#ifdef WITH_ARM_DOTPROD
    if (is_trans) {
      prepackA_m6k4_trans_int8(static_cast<int8_t*>(out),
                               static_cast<const int8_t*>(in),
                               ldin,
                               m0,
                               mmax,
                               k0,
                               kmax);
    } else {
      prepackA_m6k4_int8(static_cast<int8_t*>(out),
                         static_cast<const int8_t*>(in),
                         ldin,
                         m0,
                         mmax,
                         k0,
                         kmax);
    }
#endif
  } else {
    if (is_trans) {
      prepackA_m4k2x2_trans_int8(static_cast<int8_t*>(out),
                                 static_cast<const int8_t*>(in),
                                 ldin,
                                 m0,
                                 mmax,
                                 k0,
                                 kmax);
    } else {
      prepackA_m4k2x2_int8(static_cast<int8_t*>(out),
                           static_cast<const int8_t*>(in),
                           ldin,
                           m0,
                           mmax,
                           k0,
                           kmax);
    }
  }

#endif
}

void prepackA_int8(TensorLite* tout,
                   const TensorLite& tin,
                   int m,
                   int k,
                   int group,
                   bool is_trans,
                   ARMContext* ctx) {
  int hblock = get_hblock_int8(ctx);
  int m_roundup = ROUNDUP(m, hblock);
  // round up to 128 bits
  int kup = ROUNDUP(k, KBLOCK_INT8);
  int group_size_round_up = ((m_roundup * kup + 15) / 16) * 16;

  if (tout->numel() < group_size_round_up * group) {
    tout->Resize({1, 1, 1, group_size_round_up * group});
  }
  int lda = k;
  if (is_trans) {
    lda = m;
  }
  for (int g = 0; g < group; ++g) {
    const char* weights_group = tin.data<char>() + g * m * k;
    char* weights_trans_ptr =
        tout->mutable_data<char>() + g * group_size_round_up;
    prepackA_int8(
        weights_trans_ptr, weights_group, lda, 0, m, 0, k, is_trans, ctx);
  }
}

template <typename Dtype>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,  // NOLINT
                             const float* bias,
                             Dtype*& c_ptr0,  // NOLINT
                             Dtype*& c_ptr1,  // NOLINT
                             Dtype*& c_ptr2,  // NOLINT
                             Dtype*& c_ptr3,  // NOLINT
                             const float* scale,
                             const float32_t* alpha,
                             int is_relu,
                             int k,
                             int rem);
// clang-format off
#ifdef __aarch64__
#define GEMM_INT8_KERNEL                                                    \
  "ld1 {v0.16b}, [%[a_ptr]],#16\n"         /* load a to q0, q1 */           \
  "ld1 {v4.16b, v5.16b}, [%[b_ptr]],#32\n" /* load b to q4, q5 */           \
  "ld1 {v6.16b, v7.16b}, [%[b_ptr]],#32\n" /* load b to q6, q7 */           \
  "eor v16.16b, v8.16b, v8.16b\n"          /* set 0 to out00 */             \
  "eor v17.16b, v9.16b, v9.16b\n"          /* set 0 to out01 */             \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"    /* preload a*/                   \
  "eor v18.16b, v10.16b, v10.16b\n"        /* set 0 to out02 */             \
  "eor v19.16b, v11.16b, v11.16b\n"        /* set 0 to out03 */             \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"    /* preload b*/                   \
  "eor v20.16b, v8.16b, v8.16b\n"          /* set 0 to out10 */             \
  "eor v21.16b, v9.16b, v9.16b\n"          /* set 0 to out11 */             \
  "prfm   pldl1keep, [%[a_ptr], #128]\n"   /* preload a*/                   \
  "eor v22.16b, v10.16b, v10.16b\n"        /* set 0 to out12 */             \
  "eor v23.16b, v11.16b, v11.16b\n"        /* set 0 to out13 */             \
  "prfm   pldl1keep, [%[b_ptr], #128]\n"   /* preload b*/                   \
  "eor v24.16b, v8.16b, v8.16b\n"          /* set 0 to out20 */             \
  "eor v25.16b, v9.16b, v9.16b\n"          /* set 0 to out21 */             \
  "prfm   pldl1keep, [%[a_ptr], #192]\n"   /* preload a*/                   \
  "eor v26.16b, v10.16b, v10.16b\n"        /* set 0 to out22 */             \
  "eor v27.16b, v11.16b, v11.16b\n"        /* set 0 to out23 */             \
  "prfm   pldl1keep, [%[b_ptr], #192]\n"   /* preload b*/                   \
  "eor v28.16b, v8.16b, v8.16b\n"          /* set 0 to out30 */             \
  "eor v29.16b, v9.16b, v9.16b\n"          /* set 0 to out31 */             \
  "prfm   pldl1keep, [%[b_ptr], #256]\n"   /* preload b*/                   \
  "eor v30.16b, v10.16b, v10.16b\n"        /* set 0 to out32 */             \
  "eor v31.16b, v11.16b, v11.16b\n"        /* set 0 to out33 */             \
  "ext    v1.16b, v0.16b, v0.16b, #2\n"    /* shift left 2bytes */          \
  "ins    v1.h[3], v0.h[0]\n"              /* insert element */             \
  "ins    v1.h[7], v0.h[4]\n"              /* insert element */             \
  "rev64  v2.4s,  v0.4s\n" /* get low: 22,33,00,11; hi: 66,77,44,55 */      \
  "rev64  v3.4s,  v1.4s\n" /* get low: 33,00,11,22; hi: 77,44,55,66 */      \
  "prfm   pldl1keep, [%[b_ptr], #320]\n"   /* preload a*/                   \
  "prfm   pldl1keep, [%[b_ptr], #384]\n"   /* preload b*/                   \
  "cbz    %w[k],    3f\n" /* if k = 0, jump to remains */                   \
  /* 1st b0, b1 */                                                          \
  "smull  v8.8h,   v0.8b, v4.8b\n"         /* a0 * b0 = c00 */              \
  "smull  v12.8h,  v0.8b, v5.8b\n"         /* a0 * b1 = c01 */              \
  "smull  v9.8h,   v1.8b, v4.8b\n"         /* a1 * b0 = c10 */              \
  "smull  v13.8h,  v1.8b, v5.8b\n"         /* a1 * b1 = c11 */              \
  "smull  v10.8h,  v2.8b, v4.8b\n"         /* a2 * b0 = c20 */              \
  "smull  v14.8h,  v2.8b, v5.8b\n"         /* a2 * b1 = c21 */              \
  "smull  v11.8h,  v3.8b, v4.8b\n"         /* a3 * b0 = c30 */              \
  "smull  v15.8h,  v3.8b, v5.8b\n"         /* a3 * b1 = c31 */              \
  "subs %w[k], %w[k], #1\n" /* loop count -1 */                             \
  /* 2nd b0, b1 */                                                          \
  "smlal2  v8.8h,   v0.16b, v4.16b\n"      /* a0 * b0 = c00 */              \
  "smlal2  v12.8h,  v0.16b, v5.16b\n"      /* a0 * b1 = c01 */              \
  "smlal2  v9.8h,   v1.16b, v4.16b\n"      /* a1 * b0 = c10 */              \
  "smlal2  v13.8h,  v1.16b, v5.16b\n"      /* a1 * b1 = c11 */              \
  "smlal2  v10.8h,  v2.16b, v4.16b\n"      /* a2 * b0 = c20 */              \
  "smlal2  v14.8h,  v2.16b, v5.16b\n"      /* a2 * b1 = c21 */              \
  "smlal2  v11.8h,  v3.16b, v4.16b\n"      /* a3 * b0 = c30 */              \
  "smlal2  v15.8h,  v3.16b, v5.16b\n"      /* a3 * b1 = c31 */              \
  "beq    8f\n" /* skip main loop */                                        \
  /* main loop*/                                                            \
  "0:\n"                                                                    \
  "ld1 {v4.16b, v5.16b}, [%[b_ptr]],#32\n" /* load b to q4, q5 */           \
  /* 1st b2, b3 */                                                          \
  "sadalp  v16.4s, v8.8h\n"        /* pairwise accumulate to int32, out00 */\
  "smull  v8.8h,   v0.8b, v6.8b\n" /* a0 * b2 = c02 */                      \
  "sadalp  v20.4s, v12.8h\n"       /* pairwise accumulate to int32, out01 */\
  "smull  v12.8h,  v0.8b, v7.8b\n" /* a0 * b3 = c03 */                      \
  "sadalp  v17.4s, v9.8h\n"        /* pairwise accumulate to int32, out10 */\
  "smull  v9.8h,   v1.8b, v6.8b\n" /* a1 * b2 = c12 */                      \
  "sadalp  v21.4s, v13.8h\n"       /* pairwise accumulate to int32, out11 */\
  "smull  v13.8h,  v1.8b, v7.8b\n" /* a1 * b3 = c13 */                      \
  "sadalp  v18.4s, v10.8h\n"       /* pairwise accumulate to int32, out20 */\
  "smull  v10.8h,  v2.8b, v6.8b\n" /* a2 * b2 = c22 */                      \
  "sadalp  v22.4s, v14.8h\n"       /* pairwise accumulate to int32, out21 */\
  "smull  v14.8h,  v2.8b, v7.8b\n" /* a2 * b3 = c23 */                      \
  "sadalp  v19.4s, v11.8h\n"       /* pairwise accumulate to int32, out30 */\
  "smlal2  v8.8h,   v0.16b, v6.16b\n" /* a0 * b2 = c02 */                   \
  "smlal2  v12.8h,  v0.16b, v7.16b\n" /* a0 * b3 = c03 */                   \
  "ld1 {v0.16b}, [%[a_ptr]],#16\n"    /* load a to q0, q1 */                \
  "smull  v11.8h,  v3.8b, v6.8b\n"    /* a3 * b2 = c32 */                   \
  "sadalp  v23.4s, v15.8h\n"       /* pairwise accumulate to int32, out31 */\
  "smull  v15.8h,  v3.8b, v7.8b\n" /* a3 * b3 = c33 */                      \
  /* 2nd b2, b3 */                                                          \
  "smlal2  v9.8h,   v1.16b, v6.16b\n"   /* a1 * b2 = c12 */                 \
  "smlal2  v13.8h,  v1.16b, v7.16b\n"   /* a1 * b3 = c13 */                 \
  "smlal2  v10.8h,  v2.16b, v6.16b\n"   /* a2 * b2 = c22 */                 \
  "ext    v1.16b, v0.16b, v0.16b, #2\n" /* shift left 2bytes */             \
  "ins    v1.h[3], v0.h[0]\n"           /* insert element */                \
  "ins    v1.h[7], v0.h[4]\n"           /* insert element */                \
  "smlal2  v14.8h,  v2.16b, v7.16b\n"   /* a2 * b3 = c23 */                 \
  "smlal2  v11.8h,  v3.16b, v6.16b\n"   /* a3 * b2 = c32 */                 \
  "smlal2  v15.8h,  v3.16b, v7.16b\n"   /* a3 * b3 = c33 */                 \
  /* pre-process a*/                                                        \
  "rev64  v2.4s,  v0.4s\n" /* get low: 22,33,00,11; hi: 66,77,44,55 */      \
  "rev64  v3.4s,  v1.4s\n" /* get low: 33,00,11,22; hi: 77,44,55,66 */      \
  "ld1 {v6.16b, v7.16b}, [%[b_ptr]],#32\n" /* load b to q6, q7 */           \
  /* 1st b0, b1 */                                                          \
  "sadalp  v24.4s, v8.8h\n"        /* pairwise accumulate to int32, out02 */\
  "smull  v8.8h,   v0.8b, v4.8b\n" /* a0 * b0 = c00 */                      \
  "sadalp  v28.4s, v12.8h\n"       /* pairwise accumulate to int32, out03 */\
  "smull  v12.8h,  v0.8b, v5.8b\n" /* a0 * b1 = c01 */                      \
  "sadalp  v25.4s, v9.8h\n"        /* pairwise accumulate to int32, out12 */\
  "smull  v9.8h,   v1.8b, v4.8b\n" /* a1 * b0 = c00 */                      \
  "sadalp  v29.4s, v13.8h\n"       /* pairwise accumulate to int32, out13 */\
  "smull  v13.8h,  v1.8b, v5.8b\n" /* a1 * b1 = c01 */                      \
  "sadalp  v26.4s, v10.8h\n"       /* pairwise accumulate to int32, out22 */\
  "smull  v10.8h,  v2.8b, v4.8b\n" /* a2 * b0 = c00 */                      \
  "sadalp  v30.4s, v14.8h\n"       /* pairwise accumulate to int32, out23 */\
  "smull  v14.8h,  v2.8b, v5.8b\n" /* a2 * b1 = c01 */                      \
  "sadalp  v27.4s, v11.8h\n"       /* pairwise accumulate to int32, out32 */\
  "smull  v11.8h,  v3.8b, v4.8b\n" /* a3 * b0 = c00 */                      \
  "sadalp  v31.4s, v15.8h\n"       /* pairwise accumulate to int32, out33 */\
  "smull  v15.8h,  v3.8b, v5.8b\n" /* a3 * b1 = c01 */                      \
  "subs %w[k], %w[k], #1\n" /* loop count -1 */                             \
  /* 2nd b0, b1 */                                                          \
  "smlal2  v8.8h,   v0.16b, v4.16b\n"           /* a0 * b0 = c00 */         \
  "smlal2  v12.8h,  v0.16b, v5.16b\n"           /* a0 * b1 = c01 */         \
  "smlal2  v9.8h,   v1.16b, v4.16b\n"           /* a1 * b0 = c10 */         \
  "smlal2  v13.8h,  v1.16b, v5.16b\n"           /* a1 * b1 = c11 */         \
  "smlal2  v10.8h,  v2.16b, v4.16b\n"           /* a2 * b0 = c20 */         \
  "smlal2  v14.8h,  v2.16b, v5.16b\n"           /* a2 * b1 = c21 */         \
  "smlal2  v11.8h,  v3.16b, v4.16b\n"           /* a3 * b0 = c30 */         \
  "smlal2  v15.8h,  v3.16b, v5.16b\n"           /* a3 * b1 = c31 */         \
  "bgt 0b\n"                                    /* jump to main loop */     \
  "8:\n" /* finish main loop */                                             \
  /* 1st b2, b3 */                                                          \
  "sadalp  v16.4s, v8.8h\n"        /* pairwise accumulate to int32, out00 */\
  "smull  v8.8h,   v0.8b, v6.8b\n" /* a0 * b0 = c02 */                      \
  "sadalp  v20.4s, v12.8h\n"       /* pairwise accumulate to int32, out01 */\
  "smull  v12.8h,  v0.8b, v7.8b\n" /* a0 * b1 = c03 */                      \
  "sadalp  v17.4s, v9.8h\n"        /* pairwise accumulate to int32, out10 */\
  "smull  v9.8h,   v1.8b, v6.8b\n" /* a1 * b0 = c12 */                      \
  "sadalp  v21.4s, v13.8h\n"       /* pairwise accumulate to int32, out11 */\
  "smull  v13.8h,  v1.8b, v7.8b\n" /* a1 * b1 = c13 */                      \
  "sadalp  v18.4s, v10.8h\n"       /* pairwise accumulate to int32, out20 */\
  "smull  v10.8h,  v2.8b, v6.8b\n" /* a2 * b0 = c22 */                      \
  "sadalp  v22.4s, v14.8h\n"       /* pairwise accumulate to int32, out21 */\
  "smull  v14.8h,  v2.8b, v7.8b\n" /* a2 * b1 = c23 */                      \
  "sadalp  v19.4s, v11.8h\n"       /* pairwise accumulate to int32, out30 */\
  "smull  v11.8h,  v3.8b, v6.8b\n" /* a3 * b0 = c32 */                      \
  "sadalp  v23.4s, v15.8h\n"       /* pairwise accumulate to int32, out31 */\
  "smull  v15.8h,  v3.8b, v7.8b\n" /* a3 * b1 = c33 */ /* 2nd b2, b3 */     \
  "smlal2  v8.8h,   v0.16b, v6.16b\n"                  /* a0 * b0 = c02 */  \
  "smlal2  v12.8h,  v0.16b, v7.16b\n"                  /* a0 * b1 = c03 */  \
  "smlal2  v9.8h,   v1.16b, v6.16b\n"                  /* a1 * b0 = c12 */  \
  "smlal2  v13.8h,  v1.16b, v7.16b\n"                  /* a1 * b1 = c23 */  \
  "smlal2  v10.8h,  v2.16b, v6.16b\n"                  /* a2 * b0 = c13 */  \
  "smlal2  v14.8h,  v2.16b, v7.16b\n"                  /* a2 * b1 = c32 */  \
  "smlal2  v11.8h,  v3.16b, v6.16b\n"                  /* a3 * b0 = c22 */  \
  "smlal2  v15.8h,  v3.16b, v7.16b\n"                  /* a3 * b1 = c33 */  \
  "cbz    %w[rem],    5f\n"                            /* skip remain */    \
  "ld1 {v0.8b}, [%[a_ptr]]\n"              /* load a to q0, final */        \
  "ld1 {v4.16b, v5.16b}, [%[b_ptr]],#32\n" /* load b to q4, q5 */           \
  "ld1 {v6.16b, v7.16b}, [%[b_ptr]],#32\n" /* load b to q6, q7 */           \
  "5:\n"                                   /* no remain */                  \
  "sadalp  v24.4s, v8.8h\n"  /* pairwise accumulate to int32, out02 */      \
  "sadalp  v28.4s, v12.8h\n" /* pairwise accumulate to int32, out03 */      \
  "sadalp  v25.4s, v9.8h\n"  /* pairwise accumulate to int32, out12 */      \
  "sadalp  v29.4s, v13.8h\n" /* pairwise accumulate to int32, out13 */      \
  "sadalp  v26.4s, v10.8h\n" /* pairwise accumulate to int32, out22 */      \
  "sadalp  v30.4s, v14.8h\n" /* pairwise accumulate to int32, out23 */      \
  "sadalp  v27.4s, v11.8h\n" /* pairwise accumulate to int32, out32 */      \
  "sadalp  v31.4s, v15.8h\n" /* pairwise accumulate to int32, out33 */      \
  "3: \n"                    /* process remains */                          \
  "cbz    %w[rem],    7f\n"  /* skip remain */                              \
  /* process remain k */                                                    \
  "4: \n"                            /* remain = 1, 2 */                    \
  "ext    v1.8b, v0.8b, v0.8b, #2\n" /* shift left 2bytes */                \
  "ext    v2.8b, v0.8b, v0.8b, #4\n" /* shift left 4bytes */                \
  "ext    v3.8b, v0.8b, v0.8b, #6\n" /* shift left 6bytes */                \
  /* 1st b0, b1 */                                                          \
  "smull  v8.8h,   v0.8b, v4.8b\n"                     /* a0 * b0 = c00 */  \
  "smull  v12.8h,  v0.8b, v5.8b\n"                     /* a0 * b1 = c01 */  \
  "smull  v9.8h,   v1.8b, v4.8b\n"                     /* a1 * b0 = c10 */  \
  "smull  v13.8h,  v1.8b, v5.8b\n"                     /* a1 * b1 = c11 */  \
  "smull  v10.8h,  v2.8b, v4.8b\n"                     /* a2 * b0 = c20 */  \
  "smull  v14.8h,  v2.8b, v5.8b\n"                     /* a2 * b1 = c21 */  \
  "smull  v11.8h,  v3.8b, v4.8b\n"                     /* a3 * b0 = c30 */  \
  "smull  v15.8h,  v3.8b, v5.8b\n" /* a3 * b1 = c31 */ /* 1st b2, b3 */     \
  "sadalp  v16.4s, v8.8h\n"        /* pairwise accumulate to int32, out00 */\
  "smull  v8.8h,   v0.8b, v6.8b\n" /* a0 * b0 = c02 */                      \
  "sadalp  v20.4s, v12.8h\n"       /* pairwise accumulate to int32, out01 */\
  "smull  v12.8h,  v0.8b, v7.8b\n" /* a0 * b1 = c03 */                      \
  "sadalp  v17.4s, v9.8h\n"        /* pairwise accumulate to int32, out10 */\
  "smull  v9.8h,   v1.8b, v6.8b\n" /* a1 * b0 = c12 */                      \
  "sadalp  v21.4s, v13.8h\n"       /* pairwise accumulate to int32, out11 */\
  "smull  v13.8h,  v1.8b, v7.8b\n" /* a1 * b1 = c13 */                      \
  "sadalp  v18.4s, v10.8h\n"       /* pairwise accumulate to int32, out20 */\
  "smull  v10.8h,  v2.8b, v6.8b\n" /* a2 * b0 = c22 */                      \
  "sadalp  v22.4s, v14.8h\n"       /* pairwise accumulate to int32, out21 */\
  "smull  v14.8h,  v2.8b, v7.8b\n" /* a2 * b1 = c23 */                      \
  "sadalp  v19.4s, v11.8h\n"       /* pairwise accumulate to int32, out30 */\
  "smull  v11.8h,  v3.8b, v6.8b\n" /* a3 * b0 = c32 */                      \
  "sadalp  v23.4s, v15.8h\n"       /* pairwise accumulate to int32, out31 */\
  "smull  v15.8h,  v3.8b, v7.8b\n" /* a3 * b1 = c33 */                      \
  "sadalp  v24.4s, v8.8h\n"        /* pairwise accumulate to int32, out02 */\
  "sadalp  v28.4s, v12.8h\n"       /* pairwise accumulate to int32, out03 */\
  "sadalp  v25.4s, v9.8h\n"        /* pairwise accumulate to int32, out12 */\
  "sadalp  v29.4s, v13.8h\n"       /* pairwise accumulate to int32, out13 */\
  "sadalp  v26.4s, v10.8h\n"       /* pairwise accumulate to int32, out22 */\
  "sadalp  v30.4s, v14.8h\n"       /* pairwise accumulate to int32, out23 */\
  "sadalp  v27.4s, v11.8h\n"       /* pairwise accumulate to int32, out32 */\
  "sadalp  v31.4s, v15.8h\n"       /* pairwise accumulate to int32, out33 */\
  "7: \n"                                                                   \
  /* trans 1 */                                                             \
  "trn1   v0.4s,  v16.4s, v17.4s\n"                                         \
  "trn2   v1.4s,  v16.4s, v17.4s\n"                                         \
  "trn1   v2.4s,  v18.4s, v19.4s\n"                                         \
  "trn2   v3.4s,  v18.4s, v19.4s\n"                                         \
  "trn1   v4.4s,  v20.4s, v21.4s\n"                                         \
  "trn2   v5.4s,  v20.4s, v21.4s\n"                                         \
  "trn1   v6.4s,  v22.4s, v23.4s\n"                                         \
  "trn2   v7.4s,  v22.4s, v23.4s\n"                                         \
  "trn1   v8.4s,  v24.4s, v25.4s\n"                                         \
  "trn2   v9.4s,  v24.4s, v25.4s\n"                                         \
  "trn1  v10.4s,  v26.4s, v27.4s\n"                                         \
  "trn2  v11.4s,  v26.4s, v27.4s\n"                                         \
  "trn1  v12.4s,  v28.4s, v29.4s\n"                                         \
  "trn2  v13.4s,  v28.4s, v29.4s\n"                                         \
  "trn1  v14.4s,  v30.4s, v31.4s\n"                                         \
  "trn2  v15.4s,  v30.4s, v31.4s\n"                                         \
  /* trans 2 */                                                             \
  "trn1   v16.2d,  v0.2d, v2.2d\n"                                          \
  "trn2   v18.2d,  v0.2d, v2.2d\n"                                          \
  "trn1   v17.2d,  v1.2d, v3.2d\n"                                          \
  "trn2   v19.2d,  v1.2d, v3.2d\n"                                          \
  "trn1   v20.2d,  v4.2d, v6.2d\n"                                          \
  "trn2   v22.2d,  v4.2d, v6.2d\n"                                          \
  "trn1   v21.2d,  v5.2d, v7.2d\n"                                          \
  "trn2   v23.2d,  v5.2d, v7.2d\n"                                          \
  "trn1   v24.2d,  v8.2d, v10.2d\n"                                         \
  "trn2   v26.2d,  v8.2d, v10.2d\n"                                         \
  "trn1   v25.2d,  v9.2d, v11.2d\n"                                         \
  "trn2   v27.2d,  v9.2d, v11.2d\n"                                         \
  "trn1   v28.2d,  v12.2d, v14.2d\n"                                        \
  "trn2   v30.2d,  v12.2d, v14.2d\n"                                        \
  "trn1   v29.2d,  v13.2d, v15.2d\n"                                        \
  "trn2   v31.2d,  v13.2d, v15.2d\n"                                        \
  /* shift */                                                               \
  "ext    v17.16b, v17.16b, v17.16b, #12\n" /* circular shift left 1 */     \
  "ext    v18.16b, v18.16b, v18.16b, #8\n"  /* circular shift left 2 */     \
  "ext    v19.16b, v19.16b, v19.16b, #4\n"  /* circular shift left 3 */     \
  "ext    v21.16b, v21.16b, v21.16b, #12\n" /* circular shift left 1 */     \
  "ext    v22.16b, v22.16b, v22.16b, #8\n"  /* circular shift left 2 */     \
  "ext    v23.16b, v23.16b, v23.16b, #4\n"  /* circular shift left 3 */     \
  "ext    v25.16b, v25.16b, v25.16b, #12\n" /* circular shift left 1 */     \
  "ext    v26.16b, v26.16b, v26.16b, #8\n"  /* circular shift left 2 */     \
  "ext    v27.16b, v27.16b, v27.16b, #4\n"  /* circular shift left 3 */     \
  "ext    v29.16b, v29.16b, v29.16b, #12\n" /* circular shift left 1 */     \
  "ext    v30.16b, v30.16b, v30.16b, #8\n"  /* circular shift left 2 */     \
  "ext    v31.16b, v31.16b, v31.16b, #4\n"  /* circular shift left 3 */     \
  /* trans */                                                               \
  "trn1   v0.4s,  v16.4s, v17.4s\n" /* get a0,b0, a2,b2 */                  \
  "trn2   v1.4s,  v16.4s, v17.4s\n" /* get a1,b1, a3,b3 */                  \
  "trn1   v2.4s,  v18.4s, v19.4s\n" /* get c0,d0, c2,c2 */                  \
  "trn2   v3.4s,  v18.4s, v19.4s\n" /* get c1,d1, c3,d3 */                  \
  "trn1   v4.4s,  v20.4s, v21.4s\n"                                         \
  "trn2   v5.4s,  v20.4s, v21.4s\n"                                         \
  "trn1   v6.4s,  v22.4s, v23.4s\n"                                         \
  "trn2   v7.4s,  v22.4s, v23.4s\n"                                         \
  "trn1   v8.4s,  v24.4s, v25.4s\n"                                         \
  "trn2   v9.4s,  v24.4s, v25.4s\n"                                         \
  "trn1  v10.4s,  v26.4s, v27.4s\n"                                         \
  "trn2  v11.4s,  v26.4s, v27.4s\n"                                         \
  "trn1  v12.4s,  v28.4s, v29.4s\n"                                         \
  "trn2  v13.4s,  v28.4s, v29.4s\n"                                         \
  "trn1  v14.4s,  v30.4s, v31.4s\n"                                         \
  "trn2  v15.4s,  v30.4s, v31.4s\n" /* trans 2 */                           \
  "trn1   v16.2d,  v0.2d, v2.2d\n"  /* get a0,b0, c0,d0 */                  \
  "trn2   v24.2d,  v0.2d, v2.2d\n"  /* get a2,b2, c2,d2 */                  \
  "trn1   v20.2d,  v1.2d, v3.2d\n"  /* get a1,b1, c1,d1 */                  \
  "trn2   v28.2d,  v1.2d, v3.2d\n"  /* get a3,b3, c3,d3 */                  \
  "trn1   v17.2d,  v4.2d, v6.2d\n"                                          \
  "trn2   v25.2d,  v4.2d, v6.2d\n"                                          \
  "trn1   v21.2d,  v5.2d, v7.2d\n"                                          \
  "trn2   v29.2d,  v5.2d, v7.2d\n"                                          \
  "trn1   v18.2d,  v8.2d, v10.2d\n"                                         \
  "trn2   v26.2d,  v8.2d, v10.2d\n"                                         \
  "trn1   v22.2d,  v9.2d, v11.2d\n"                                         \
  "trn2   v30.2d,  v9.2d, v11.2d\n"                                         \
  "trn1   v19.2d,  v12.2d, v14.2d\n"                                        \
  "trn2   v27.2d,  v12.2d, v14.2d\n"                                        \
  "trn1   v23.2d,  v13.2d, v15.2d\n"                                        \
  "trn2   v31.2d,  v13.2d, v15.2d\n"

#define GEMM_INT8_RELU                             \
  /* do relu */                                    \
  "cmp    %w[is_relu],    #0\n"    /* skip relu */ \
  "beq   9f                     \n"   /* no act end */ \
  "cmp    %w[is_relu],    #1\n"    /* skip relu */ \
  "movi   v0.4s, #0\n"             /* for relu */  \
  "bne   10f                     \n"   /* other act */ \
  "fmax   v16.4s, v16.4s, v0.4s\n" /* relu */      \
  "fmax   v17.4s, v17.4s, v0.4s\n" /* relu */      \
  "fmax   v18.4s, v18.4s, v0.4s\n" /* relu */      \
  "fmax   v19.4s, v19.4s, v0.4s\n" /* relu */      \
  "fmax   v20.4s, v20.4s, v0.4s\n" /* relu */      \
  "fmax   v21.4s, v21.4s, v0.4s\n" /* relu */      \
  "fmax   v22.4s, v22.4s, v0.4s\n" /* relu */      \
  "fmax   v23.4s, v23.4s, v0.4s\n" /* relu */      \
  "fmax   v24.4s, v24.4s, v0.4s\n" /* relu */      \
  "fmax   v25.4s, v25.4s, v0.4s\n" /* relu */      \
  "fmax   v26.4s, v26.4s, v0.4s\n" /* relu */      \
  "fmax   v27.4s, v27.4s, v0.4s\n" /* relu */      \
  "fmax   v28.4s, v28.4s, v0.4s\n" /* relu */      \
  "fmax   v29.4s, v29.4s, v0.4s\n" /* relu */      \
  "fmax   v30.4s, v30.4s, v0.4s\n" /* relu */      \
  "fmax   v31.4s, v31.4s, v0.4s\n" /* relu */      \
  "b      9f                    \n"   /* relu end */

#define GEMM_INT8_RELU6                             \
  /* do relu6 */                                     \
  "10: \n"                                           \
  "cmp   %w[is_relu],  #2       \n"   /* check relu6 */ \
  "bne   11f                     \n"   /* no act end */ \
  "fmax   v16.4s, v16.4s, v0.4s\n" /* relu */      \
  "fmax   v17.4s, v17.4s, v0.4s\n" /* relu */      \
  "fmax   v18.4s, v18.4s, v0.4s\n" /* relu */      \
  "fmax   v19.4s, v19.4s, v0.4s\n" /* relu */      \
  "fmax   v20.4s, v20.4s, v0.4s\n" /* relu */      \
  "ld1    {v1.4s}, [%[alpha]]    \n"    /* relu6 alpha */ \
  "fmax   v21.4s, v21.4s, v0.4s\n" /* relu */      \
  "fmax   v22.4s, v22.4s, v0.4s\n" /* relu */      \
  "fmax   v23.4s, v23.4s, v0.4s\n" /* relu */      \
  "fmax   v24.4s, v24.4s, v0.4s\n" /* relu */      \
  "fmax   v25.4s, v25.4s, v0.4s\n" /* relu */      \
  "fmax   v26.4s, v26.4s, v0.4s\n" /* relu */      \
  "fmax   v27.4s, v27.4s, v0.4s\n" /* relu */      \
  "fmax   v28.4s, v28.4s, v0.4s\n" /* relu */      \
  "fmax   v29.4s, v29.4s, v0.4s\n" /* relu */      \
  "fmax   v30.4s, v30.4s, v0.4s\n" /* relu */      \
  "fmax   v31.4s, v31.4s, v0.4s\n" /* relu */      \
  "fmin   v16.4s, v16.4s, v1.4s\n" /* relu6 */     \
  "fmin   v17.4s, v17.4s, v1.4s\n" /* relu6 */     \
  "fmin   v18.4s, v18.4s, v1.4s\n" /* relu6 */     \
  "fmin   v19.4s, v19.4s, v1.4s\n" /* relu6 */     \
  "fmin   v20.4s, v20.4s, v1.4s\n" /* relu6 */     \
  "fmin   v21.4s, v21.4s, v1.4s\n" /* relu6 */     \
  "fmin   v22.4s, v22.4s, v1.4s\n" /* relu6 */     \
  "fmin   v23.4s, v23.4s, v1.4s\n" /* relu6 */     \
  "fmin   v24.4s, v24.4s, v1.4s\n" /* relu6 */     \
  "fmin   v25.4s, v25.4s, v1.4s\n" /* relu6 */     \
  "fmin   v26.4s, v26.4s, v1.4s\n" /* relu6 */     \
  "fmin   v27.4s, v27.4s, v1.4s\n" /* relu6 */     \
  "fmin   v28.4s, v28.4s, v1.4s\n" /* relu6 */     \
  "fmin   v29.4s, v29.4s, v1.4s\n" /* relu6 */     \
  "fmin   v30.4s, v30.4s, v1.4s\n" /* relu6 */     \
  "fmin   v31.4s, v31.4s, v1.4s\n" /* relu6 */     \
  "b      9f                    \n"   /* relu end */

#define GEMM_INT8_LEAKY_RELU                       \
  /* do leakyrelu */                               \
  "11: \n"                                         \
  "cmp   %w[is_relu],  #3       \n"   /* check relu6 */ \
  "bne   12f                     \n"   /* no act end */ \
  "ld1    {v1.4s},  [%[alpha]]        \n" /* leakey relu alpha */ \
  "fcmge  v2.4s,    v16.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v3.4s,    v16.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "fcmge  v4.4s,    v17.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v17.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v18.4s,   v0.4s   \n" /* vcgeq_f32 */   \
  "fmul   v7.4s,    v18.4s,   v1.4s   \n" /* vmulq_f32 */   \
  "fcmge  v8.4s,    v19.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v9.4s,    v19.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "bif    v16.16b,   v3.16b,   v2.16b  \n" /* choose*/      \
  "bif    v17.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v18.16b,  v7.16b,   v6.16b  \n" /* choose*/       \
  "bif    v19.16b,  v9.16b,   v8.16b  \n" /* choose*/       \
  "fcmge  v2.4s,    v20.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v3.4s,    v20.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "fcmge  v4.4s,    v21.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v21.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v22.4s,   v0.4s   \n" /* vcgeq_f32 */   \
  "fmul   v7.4s,    v22.4s,   v1.4s   \n" /* vmulq_f32 */   \
  "fcmge  v8.4s,    v23.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v9.4s,    v23.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "bif    v20.16b,   v3.16b,   v2.16b  \n" /* choose*/      \
  "bif    v21.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v22.16b,  v7.16b,   v6.16b  \n" /* choose*/       \
  "bif    v23.16b,  v9.16b,   v8.16b  \n" /* choose*/       \
  "fcmge  v2.4s,    v24.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v3.4s,    v24.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "fcmge  v4.4s,    v25.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v25.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v26.4s,   v0.4s   \n" /* vcgeq_f32 */   \
  "fmul   v7.4s,    v26.4s,   v1.4s   \n" /* vmulq_f32 */   \
  "fcmge  v8.4s,    v27.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v9.4s,    v27.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "bif    v24.16b,   v3.16b,   v2.16b  \n" /* choose*/      \
  "bif    v25.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v26.16b,  v7.16b,   v6.16b  \n" /* choose*/       \
  "bif    v27.16b,  v9.16b,   v8.16b  \n" /* choose*/       \
  "fcmge  v2.4s,    v28.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v3.4s,    v28.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "fcmge  v4.4s,    v29.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v29.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v30.4s,   v0.4s   \n" /* vcgeq_f32 */   \
  "fmul   v7.4s,    v30.4s,   v1.4s   \n" /* vmulq_f32 */   \
  "fcmge  v8.4s,    v31.4s,    v0.4s   \n" /* vcgeq_f32 */  \
  "fmul   v9.4s,    v31.4s,    v1.4s   \n" /* vmulq_f32 */  \
  "bif    v28.16b,   v3.16b,   v2.16b  \n" /* choose*/      \
  "bif    v29.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v30.16b,  v7.16b,   v6.16b  \n" /* choose*/       \
  "bif    v31.16b,  v9.16b,   v8.16b  \n" /* choose*/       \
  "b 9f\n"

#define GEMM_INT8_HARD_SWISH                       \
  /* do hardswsih */                               \
  "12: \n"                                         \
  "ldr    q2,       [%[alpha], #16]    \n" /* hard_swish offset*/ \
  "ld1    {v1.4s},  [%[alpha]]         \n" /* hard_swish alpha */ \
  "ldr    q3,       [%[alpha], #32]    \n" /* hard_swish threshold */\
  "fadd   v4.4s,    v16.4s,   v2.4s    \n"         \
  "fmul   v16.4s,   v16.4s,   v1.4s    \n"         \
  "fadd   v5.4s,    v17.4s,   v2.4s    \n"         \
  "fmul   v17.4s,   v17.4s,   v1.4s    \n"         \
  "fadd   v6.4s,    v18.4s,   v2.4s    \n"         \
  "fmul   v18.4s,   v18.4s,   v1.4s    \n"         \
  "fadd   v7.4s,    v19.4s,   v2.4s    \n"         \
  "fmul   v19.4s,   v19.4s,   v1.4s    \n"         \
  "fmax   v4.4s,    v4.4s,    v0.4s    \n"         \
  "fmax   v5.4s,    v5.4s,    v0.4s    \n"         \
  "fmax   v6.4s,    v6.4s,    v0.4s    \n"         \
  "fmax   v7.4s,    v7.4s,    v0.4s    \n"         \
  "fmin   v4.4s,    v4.4s,    v3.4s    \n"         \
  "fmin   v5.4s,    v5.4s,    v3.4s    \n"         \
  "fmin   v6.4s,    v6.4s,    v3.4s    \n"         \
  "fmin   v7.4s,    v7.4s,    v3.4s    \n"         \
  "fmul   v16.4s,   v16.4s,   v4.4s    \n"         \
  "fmul   v17.4s,   v17.4s,   v5.4s    \n"         \
  "fmul   v18.4s,   v18.4s,   v6.4s    \n"         \
  "fmul   v19.4s,   v19.4s,   v7.4s    \n"         \
  "fadd   v4.4s,    v20.4s,   v2.4s    \n"         \
  "fmul   v20.4s,   v20.4s,   v1.4s    \n"         \
  "fadd   v5.4s,    v21.4s,   v2.4s    \n"         \
  "fmul   v21.4s,   v21.4s,   v1.4s    \n"         \
  "fadd   v6.4s,    v22.4s,   v2.4s    \n"         \
  "fmul   v22.4s,   v22.4s,   v1.4s    \n"         \
  "fadd   v7.4s,    v23.4s,   v2.4s    \n"         \
  "fmul   v23.4s,   v23.4s,   v1.4s    \n"         \
  "fmax   v4.4s,    v4.4s,    v0.4s    \n"         \
  "fmax   v5.4s,    v5.4s,    v0.4s    \n"         \
  "fmax   v6.4s,    v6.4s,    v0.4s    \n"         \
  "fmax   v7.4s,    v7.4s,    v0.4s    \n"         \
  "fmin   v4.4s,    v4.4s,    v3.4s    \n"         \
  "fmin   v5.4s,    v5.4s,    v3.4s    \n"         \
  "fmin   v6.4s,    v6.4s,    v3.4s    \n"         \
  "fmin   v7.4s,    v7.4s,    v3.4s    \n"         \
  "fmul   v20.4s,   v20.4s,   v4.4s    \n"         \
  "fmul   v21.4s,   v21.4s,   v5.4s    \n"         \
  "fmul   v22.4s,   v22.4s,   v6.4s    \n"         \
  "fmul   v23.4s,   v23.4s,   v7.4s    \n"         \
  "fadd   v4.4s,    v24.4s,   v2.4s    \n"         \
  "fmul   v24.4s,   v24.4s,   v1.4s    \n"         \
  "fadd   v5.4s,    v25.4s,   v2.4s    \n"         \
  "fmul   v25.4s,   v25.4s,   v1.4s    \n"         \
  "fadd   v6.4s,    v26.4s,   v2.4s    \n"         \
  "fmul   v26.4s,   v26.4s,   v1.4s    \n"         \
  "fadd   v7.4s,    v27.4s,   v2.4s    \n"         \
  "fmul   v27.4s,   v27.4s,   v1.4s    \n"         \
  "fmax   v4.4s,    v4.4s,    v0.4s    \n"         \
  "fmax   v5.4s,    v5.4s,    v0.4s    \n"         \
  "fmax   v6.4s,    v6.4s,    v0.4s    \n"         \
  "fmax   v7.4s,    v7.4s,    v0.4s    \n"         \
  "fmin   v4.4s,    v4.4s,    v3.4s    \n"         \
  "fmin   v5.4s,    v5.4s,    v3.4s    \n"         \
  "fmin   v6.4s,    v6.4s,    v3.4s    \n"         \
  "fmin   v7.4s,    v7.4s,    v3.4s    \n"         \
  "fmul   v24.4s,   v24.4s,   v4.4s    \n"         \
  "fmul   v25.4s,   v25.4s,   v5.4s    \n"         \
  "fmul   v26.4s,   v26.4s,   v6.4s    \n"         \
  "fmul   v27.4s,   v27.4s,   v7.4s    \n"         \
  "fadd   v4.4s,    v28.4s,   v2.4s    \n"         \
  "fmul   v28.4s,   v28.4s,   v1.4s    \n"         \
  "fadd   v5.4s,    v29.4s,   v2.4s    \n"         \
  "fmul   v29.4s,   v29.4s,   v1.4s    \n"         \
  "fadd   v6.4s,    v30.4s,   v2.4s    \n"         \
  "fmul   v30.4s,   v30.4s,   v1.4s    \n"         \
  "fadd   v7.4s,    v31.4s,   v2.4s    \n"         \
  "fmul   v31.4s,   v31.4s,   v1.4s    \n"         \
  "fmax   v4.4s,    v4.4s,    v0.4s    \n"         \
  "fmax   v5.4s,    v5.4s,    v0.4s    \n"         \
  "fmax   v6.4s,    v6.4s,    v0.4s    \n"         \
  "fmax   v7.4s,    v7.4s,    v0.4s    \n"         \
  "fmin   v4.4s,    v4.4s,    v3.4s    \n"         \
  "fmin   v5.4s,    v5.4s,    v3.4s    \n"         \
  "fmin   v6.4s,    v6.4s,    v3.4s    \n"         \
  "fmin   v7.4s,    v7.4s,    v3.4s    \n"         \
  "fmul   v28.4s,   v28.4s,   v4.4s    \n"         \
  "fmul   v29.4s,   v29.4s,   v5.4s    \n"         \
  "fmul   v30.4s,   v30.4s,   v6.4s    \n"         \
  "fmul   v31.4s,   v31.4s,   v7.4s    \n"         \
  "9:\n"

#define GEMM_TRANS_INT32_TO_FP32                                      \
  "ldr    q14, [%[bias]]\n"  /* load scale */                         \
  "ldr    q15, [%[scale]]\n" /* load scale */                         \
  "scvtf  v0.4s , v16.4s\n"  /*  00, convert to fp32 */               \
  "scvtf  v1.4s , v17.4s\n"  /*  01, convert to fp32 */               \
  "scvtf  v2.4s , v18.4s\n"  /*  02, convert to fp32 */               \
  "scvtf  v3.4s , v19.4s\n"  /*  03, convert to fp32 */               \
  "scvtf  v4.4s , v20.4s\n"  /*  10, convert to fp32 */               \
  "scvtf  v5.4s , v21.4s\n"  /*  11, convert to fp32 */               \
  "scvtf  v6.4s , v22.4s\n"  /*  12, convert to fp32 */               \
  "scvtf  v7.4s , v23.4s\n"  /*  13, convert to fp32 */               \
  /* add bias */                                                      \
  "dup    v16.4s, v14.s[0]\n"                                         \
  "dup    v17.4s, v14.s[0]\n"                                         \
  "dup    v18.4s, v14.s[0]\n"                                         \
  "dup    v19.4s, v14.s[0]\n"                                         \
  "dup    v20.4s, v14.s[1]\n"                                         \
  "dup    v21.4s, v14.s[1]\n"                                         \
  "dup    v22.4s, v14.s[1]\n"                                         \
  "dup    v23.4s, v14.s[1]\n"                                         \
  "fmla   v16.4s, v0.4s, v15.s[0]\n" /*  00, mul scale */             \
  "fmla   v17.4s, v1.4s, v15.s[0]\n" /*  01, mul scale */             \
  "fmla   v18.4s, v2.4s, v15.s[0]\n" /*  02, mul scale */             \
  "fmla   v19.4s, v3.4s, v15.s[0]\n" /*  03, mul scale */             \
  "fmla   v20.4s, v4.4s, v15.s[1]\n" /*  10, mul scale */             \
  "fmla   v21.4s, v5.4s, v15.s[1]\n" /*  11, mul scale */             \
  "fmla   v22.4s, v6.4s, v15.s[1]\n" /*  12, mul scale */             \
  "fmla   v23.4s, v7.4s, v15.s[1]\n" /*  13, mul scale */             \
  "scvtf  v0.4s , v24.4s\n"          /*  20, convert to fp32 */       \
  "scvtf  v1.4s , v25.4s\n"          /*  21, convert to fp32 */       \
  "scvtf  v2.4s , v26.4s\n"          /*  22, convert to fp32 */       \
  "scvtf  v3.4s , v27.4s\n"          /*  23, convert to fp32 */       \
  "scvtf  v4.4s , v28.4s\n"          /*  30, convert to fp32 */       \
  "scvtf  v5.4s , v29.4s\n"          /*  31, convert to fp32 */       \
  "scvtf  v6.4s , v30.4s\n"          /*  32, convert to fp32 */       \
  "scvtf  v7.4s , v31.4s\n"          /*  33, convert to fp32 */       \
  "dup    v24.4s, v14.s[2]\n"                                         \
  "dup    v25.4s, v14.s[2]\n"                                         \
  "dup    v26.4s, v14.s[2]\n"                                         \
  "dup    v27.4s, v14.s[2]\n"                                         \
  "dup    v28.4s, v14.s[3]\n"                                         \
  "dup    v29.4s, v14.s[3]\n"                                         \
  "dup    v30.4s, v14.s[3]\n"                                         \
  "dup    v31.4s, v14.s[3]\n"                                         \
  "fmla   v24.4s, v0.4s, v15.s[2]\n" /*  20, mul scale */             \
  "fmla   v25.4s, v1.4s, v15.s[2]\n" /*  21, mul scale */             \
  "fmla   v26.4s, v2.4s, v15.s[2]\n" /*  22, mul scale */             \
  "fmla   v27.4s, v3.4s, v15.s[2]\n" /*  23, mul scale */             \
  "fmla   v28.4s, v4.4s, v15.s[3]\n" /*  30, mul scale */             \
  "fmla   v29.4s, v5.4s, v15.s[3]\n" /*  31, mul scale */             \
  "fmla   v30.4s, v6.4s, v15.s[3]\n" /*  32, mul scale */             \
  "fmla   v31.4s, v7.4s, v15.s[3]\n" /*  33, mul scale */

#define GEMM_INT8_FP32_OUT                \
  GEMM_TRANS_INT32_TO_FP32                \
  GEMM_INT8_RELU                          \
  GEMM_INT8_RELU6                         \
  GEMM_INT8_LEAKY_RELU                    \
  GEMM_INT8_HARD_SWISH                    \
  /* store result */                      \
  "stp    q16, q17,   [%[c_ptr0]], #32\n" \
  "stp    q18, q19,   [%[c_ptr0]], #32\n" \
  "stp    q20, q21,   [%[c_ptr1]], #32\n" \
  "stp    q22, q23,   [%[c_ptr1]], #32\n" \
  "stp    q24, q25,   [%[c_ptr2]], #32\n" \
  "stp    q26, q27,   [%[c_ptr2]], #32\n" \
  "stp    q28, q29,   [%[c_ptr3]], #32\n" \
  "stp    q30, q31,   [%[c_ptr3]], #32\n"

#define GEMM_INT8_INT8_OUT                                         \
  GEMM_TRANS_INT32_TO_FP32                                         \
  GEMM_INT8_RELU                                                   \
  GEMM_INT8_RELU6                                                  \
  GEMM_INT8_LEAKY_RELU                                             \
  GEMM_INT8_HARD_SWISH                                             \
  "ld1    {v8.4s},   [%[vmax]] \n"          /* v8 = -127 */        \
  /* data >= -127 */                                               \
  "fcmge v0.4s, v16.4s, v8.4s\n"                                   \
  "fcmge v1.4s, v17.4s, v8.4s\n"                                   \
  "fcmge v2.4s, v18.4s, v8.4s\n"                                   \
  "fcmge v3.4s, v19.4s, v8.4s\n"                                   \
  "fcmge v4.4s, v20.4s, v8.4s\n"                                   \
  "fcmge v5.4s, v21.4s, v8.4s\n"                                   \
  "fcmge v6.4s, v22.4s, v8.4s\n"                                   \
  "fcmge v7.4s, v23.4s, v8.4s\n"                                   \
  /* choose data */                                                \
  "bif v16.16b, v8.16b, v0.16b            \n"                      \
  "bif v17.16b, v8.16b, v1.16b            \n"                      \
  "bif v18.16b, v8.16b, v2.16b            \n"                      \
  "bif v19.16b, v8.16b, v3.16b            \n"                      \
  "bif v20.16b, v8.16b, v4.16b            \n"                      \
  "bif v21.16b, v8.16b, v5.16b            \n"                      \
  "bif v22.16b, v8.16b, v6.16b            \n"                      \
  "bif v23.16b, v8.16b, v7.16b            \n"                      \
  "fcvtas v0.4s, v16.4s\n"        /*  00, cvt to int */            \
  "fcvtas v1.4s, v17.4s\n"        /*  01, cvt to int */            \
  "fcvtas v2.4s, v18.4s\n"        /*  02, cvt to int */            \
  "fcvtas v3.4s, v19.4s\n"        /*  03, cvt to int */            \
  "fcvtas v4.4s, v20.4s\n"        /*  10, cvt to int */            \
  "fcvtas v5.4s, v21.4s\n"        /*  11, cvt to int */            \
  "fcvtas v6.4s, v22.4s\n"        /*  12, cvt to int */            \
  "fcvtas v7.4s, v23.4s\n"        /*  13, cvt to int */            \
  /* data >= -127 */                                               \
  "fcmge v16.4s, v24.4s, v8.4s\n"                                   \
  "fcmge v17.4s, v25.4s, v8.4s\n"                                   \
  "fcmge v18.4s, v26.4s, v8.4s\n"                                   \
  "fcmge v19.4s, v27.4s, v8.4s\n"                                   \
  "fcmge v20.4s, v28.4s, v8.4s\n"                                   \
  "fcmge v21.4s, v29.4s, v8.4s\n"                                   \
  "fcmge v22.4s, v30.4s, v8.4s\n"                                   \
  "fcmge v23.4s, v31.4s, v8.4s\n"                                   \
  /* choose data */                                                \
  "bif v24.16b, v8.16b, v16.16b\n"                                  \
  "bif v25.16b, v8.16b, v17.16b\n"                                  \
  "bif v26.16b, v8.16b, v18.16b\n"                                  \
  "bif v27.16b, v8.16b, v19.16b\n"                                  \
  "bif v28.16b, v8.16b, v20.16b\n"                                  \
  "bif v29.16b, v8.16b, v21.16b\n"                                  \
  "bif v30.16b, v8.16b, v22.16b\n"                                  \
  "bif v31.16b, v8.16b, v23.16b\n"                                  \
  "sqxtn  v16.4h, v0.4s\n"        /*  00, cvt int32 to int16 */    \
  "fcvtas v8.4s, v24.4s\n"        /*  20, cvt to int */            \
  "sqxtn2 v16.8h, v1.4s\n"        /*  01, cvt int32 to int16 */    \
  "fcvtas v9.4s, v25.4s\n"        /*  21, cvt to int */            \
  "sqxtn  v17.4h, v2.4s\n"        /*  02, cvt int32 to int16 */    \
  "fcvtas v10.4s, v26.4s\n"       /*  22, cvt to int */            \
  "sqxtn2 v17.8h, v3.4s\n"        /*  03, cvt int32 to int16 */    \
  "fcvtas v11.4s, v27.4s\n"       /*  23, cvt to int */            \
  "sqxtn  v18.4h, v4.4s\n"        /*  10, cvt int32 to int16 */    \
  "fcvtas v12.4s, v28.4s\n"       /*  30, cvt to int */            \
  "sqxtn2 v18.8h, v5.4s\n"        /*  11, cvt int32 to int16 */    \
  "fcvtas v13.4s, v29.4s\n"       /*  31, cvt to int */            \
  "sqxtn  v19.4h, v6.4s\n"        /*  12, cvt int32 to int16 */    \
  "fcvtas v14.4s, v30.4s\n"       /*  32, cvt to int */            \
  "sqxtn2 v19.8h, v7.4s\n"        /*  13, cvt int32 to int16 */    \
  "fcvtas v15.4s, v31.4s\n"       /*  33, cvt to int */            \
  "sqxtn  v0.8b, v16.8h\n"        /*  00, 01, cvt int16 to int8 */ \
  "sqxtn2 v0.16b, v17.8h\n"       /*  02, 03, cvt int16 to int8 */ \
  "sqxtn  v1.8b, v18.8h\n"        /*  10, 11, cvt int16 to int8 */ \
  "sqxtn2 v1.16b, v19.8h\n"       /*  12, 13, cvt int16 to int8 */ \
  "sqxtn  v20.4h, v8.4s\n"        /*  20, cvt int32 to int16 */    \
  "sqxtn2 v20.8h, v9.4s\n"        /*  21, cvt int32 to int16 */    \
  "sqxtn  v21.4h, v10.4s\n"       /*  22, cvt int32 to int16 */    \
  "sqxtn2 v21.8h, v11.4s\n"       /*  23, cvt int32 to int16 */    \
  "sqxtn  v22.4h, v12.4s\n"       /*  30, cvt int32 to int16 */    \
  "sqxtn2 v22.8h, v13.4s\n"       /*  31, cvt int32 to int16 */    \
  "sqxtn  v23.4h, v14.4s\n"       /*  32, cvt int32 to int16 */    \
  "sqxtn2 v23.8h, v15.4s\n"       /*  33, cvt int32 to int16 */    \
  "sqxtn  v2.8b, v20.8h\n"        /*  20, 21, cvt int16 to int8 */ \
  "sqxtn2 v2.16b, v21.8h\n"       /*  22, 23, cvt int16 to int8 */ \
  "sqxtn  v3.8b, v22.8h\n"        /*  30, 31, cvt int16 to int8 */ \
  "sqxtn2 v3.16b, v23.8h\n"       /*  32, 33, cvt int16 to int8 */ \
  "str    q0, [%[c_ptr0]], #16\n" /*  write r0 */                  \
  "str    q1, [%[c_ptr1]], #16\n" /*  write r1 */                  \
  "str    q2, [%[c_ptr2]], #16\n" /*  write r2 */                  \
  "str    q3, [%[c_ptr3]], #16\n" /*  write r3 */


#define GEMM_INT8_INT32_OUT               \
  /* store result */                      \
  "stp    q16, q17,   [%[c_ptr0]], #32\n" \
  "stp    q18, q19,   [%[c_ptr0]], #32\n" \
  "stp    q20, q21,   [%[c_ptr1]], #32\n" \
  "stp    q22, q23,   [%[c_ptr1]], #32\n" \
  "stp    q24, q25,   [%[c_ptr2]], #32\n" \
  "stp    q26, q27,   [%[c_ptr2]], #32\n" \
  "stp    q28, q29,   [%[c_ptr3]], #32\n" \
  "stp    q30, q31,   [%[c_ptr3]], #32\n"
// clang-format on

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,  // NOLINT
                             const float* bias,
                             float32_t*& c_ptr0,  // NOLINT
                             float32_t*& c_ptr1,  // NOLINT
                             float32_t*& c_ptr2,  // NOLINT
                             float32_t*& c_ptr3,  // NOLINT
                             const float32_t* scale,
                             const float32_t* alpha,
                             int is_relu,
                             int k,
                             int rem) {
  // clang-format off
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_FP32_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [k] "+r"(k)
               : [is_relu] "r"(is_relu),
                 [alpha] "r"(alpha),
                 [bias] "r"(bias),
                 [rem] "r"(rem),
                 [scale] "r"(scale)
               : "v0","v1","v2","v3","v4","v5","v6","v7","v8",
                 "v9","v10","v11","v12","v13","v14",
                 "v15","v16","v17","v18","v19","v20",
                 "v21","v22","v23","v24","v25","v26",
                 "v27","v28","v29","v30","v31","cc", "memory");
  // clang-format on
}

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,  // NOLINT
                             const float* bias,
                             int8_t*& c_ptr0,  // NOLINT
                             int8_t*& c_ptr1,  // NOLINT
                             int8_t*& c_ptr2,  // NOLINT
                             int8_t*& c_ptr3,  // NOLINT
                             const float32_t* scale,
                             const float32_t* alpha,
                             int is_relu,
                             int k,
                             int rem) {
  // clang-format off
  float vmax[4] = {-127.0, -127.0, -127.0, -127.0};
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_INT8_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [k] "+r"(k)
               : [is_relu] "r"(is_relu),
                 [alpha] "r"(alpha),
                 [bias] "r"(bias),
                 [rem] "r"(rem),
                 [scale] "r"(scale),
                 [vmax] "r"(vmax)
               : "v0","v1","v2","v3","v4","v5","v6","v7",
                 "v8","v9","v10","v11","v12",
                 "v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22",
                 "v23","v24","v25","v26","v27",
                 "v28","v29","v30","v31","cc", "memory");
  // clang-format on
}

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,  // NOLINT
                             const float* bias,
                             int32_t*& c_ptr0,  // NOLINT
                             int32_t*& c_ptr1,  // NOLINT
                             int32_t*& c_ptr2,  // NOLINT
                             int32_t*& c_ptr3,  // NOLINT
                             const float32_t* scale,
                             const float32_t* alpha,
                             int is_relu,
                             int k,
                             int rem) {
  // clang-format off
  float vmax[4] = {-127.0, -127.0, -127.0, -127.0};
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_INT32_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [k] "+r"(k)
               : [is_relu] "r"(is_relu),
                 [alpha] "r"(alpha),
                 [bias] "r"(bias),
                 [rem] "r"(rem),
                 [scale] "r"(scale),
                 [vmax] "r"(vmax)
               : "v0","v1","v2","v3","v4","v5","v6","v7",
                 "v8","v9","v10","v11","v12",
                 "v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22",
                 "v23","v24","v25","v26","v27",
                 "v28","v29","v30","v31","cc", "memory");
  // clang-format on
}

#ifdef WITH_ARM_DOTPROD

template <typename Dtype>
inline void gemm_sdot_int8_kernel_8x4(const int8_t* a_ptr,
                                      const int8_t*& b_ptr,  // NOLINT
                                      const float* bias,
                                      Dtype*& c_ptr0,  // NOLINT
                                      Dtype*& c_ptr1,  // NOLINT
                                      Dtype*& c_ptr2,  // NOLINT
                                      Dtype*& c_ptr3,  // NOLINT
                                      Dtype*& c_ptr4,  // NOLINT
                                      Dtype*& c_ptr5,  // NOLINT
                                      Dtype*& c_ptr6,  // NOLINT
                                      Dtype*& c_ptr7,  // NOLINT
                                      const float32_t* scale,
                                      const float32_t* alpha,
                                      int is_relu,
                                      int k,
                                      int rem);
template <typename Dtype>
inline void gemm_sdot_int8_kernel_8x8(const int8_t* a_ptr,
                                      const int8_t*& b_ptr,  // NOLINT
                                      const float* bias,
                                      Dtype*& c_ptr0,  // NOLINT
                                      Dtype*& c_ptr1,  // NOLINT
                                      Dtype*& c_ptr2,  // NOLINT
                                      Dtype*& c_ptr3,  // NOLINT
                                      Dtype*& c_ptr4,  // NOLINT
                                      Dtype*& c_ptr5,  // NOLINT
                                      Dtype*& c_ptr6,  // NOLINT
                                      Dtype*& c_ptr7,  // NOLINT
                                      const float32_t* scale,
                                      const float32_t* alpha,
                                      int is_relu,
                                      int k,
                                      int rem);
template <typename Dtype>
inline void gemm_sdot_int8_kernel(const int8_t* a_ptr,
                                  const int8_t*& b_ptr,  // NOLINT
                                  const float* bias,
                                  Dtype*& c_ptr0,  // NOLINT
                                  Dtype*& c_ptr1,  // NOLINT
                                  Dtype*& c_ptr2,  // NOLINT
                                  Dtype*& c_ptr3,  // NOLINT
                                  Dtype*& c_ptr4,  // NOLINT
                                  Dtype*& c_ptr5,  // NOLINT
                                  Dtype*& c_ptr6,  // NOLINT
                                  Dtype*& c_ptr7,  // NOLINT
                                  const float32_t* scale,
                                  const float32_t* alpha,
                                  int is_relu,
                                  int k,
                                  int rem);
#if 0
// clang-format off
#define GEMM_SDOT_INT8_KERNEL                                              \
  "ldp    q0, q1, [%[a_ptr]], #32\n"     /* load a00,a01 to q0, q1*/       \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b0, b1 to q4, q5*/        \
  "eor    v8.16b,  v8.16b, v8.16b\n"     /* out0 = 0 */                    \
  "eor    v9.16b,  v9.16b, v9.16b\n"     /* out1 = 0 */                    \
  "eor    v10.16b,  v10.16b, v10.16b\n"  /* out2 = 0 */                    \
  "eor    v11.16b,  v11.16b, v11.16b\n"  /* out3 = 0 */                    \
  "eor    v12.16b,  v12.16b, v12.16b\n"  /* out4 = 0 */                    \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"  /* preload b*/                    \
  "eor    v13.16b,  v13.16b, v13.16b\n"  /* out5 = 0 */                    \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/                    \
  "eor    v14.16b,  v14.16b, v14.16b\n"  /* out6 = 0 */                    \
  "prfm   pldl1keep, [%[b_ptr], #128]\n" /* preload b*/                    \
  "eor    v15.16b,  v15.16b, v15.16b\n"  /* out7 = 0 */                    \
  "prfm   pldl1keep, [%[a_ptr], #128]\n" /* preload a*/                    \
  "eor    v16.16b,  v16.16b, v16.16b\n"  /* out8 = 0 */                    \
  "prfm   pldl1keep, [%[b_ptr], #192]\n" /* preload b*/                    \
  "eor    v17.16b,  v17.16b, v17.16b\n"  /* out9 = 0 */                    \
  "prfm   pldl1keep, [%[b_ptr], #256]\n" /* preload b*/                    \
  "eor    v18.16b,  v18.16b, v18.16b\n"  /* out10 = 0 */                   \
  "prfm   pldl1keep, [%[a_ptr], #192]\n" /* preload a*/                    \
  "eor    v19.16b,  v19.16b, v19.16b\n"  /* out11 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #320]\n" /* preload b*/                    \
  "eor    v20.16b,  v20.16b, v20.16b\n"  /* out12 = 0 */                   \
  "prfm   pldl1keep, [%[a_ptr], #256]\n" /* preload a*/                    \
  "eor    v21.16b,  v21.16b, v21.16b\n"  /* out13 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #384]\n" /* preload b*/                    \
  "eor    v22.16b,  v22.16b, v22.16b\n"  /* out14 = 0 */                   \
  "eor    v23.16b,  v23.16b, v23.16b\n"  /* out15 = 0 */                   \
  "eor    v24.16b,  v24.16b, v24.16b\n"  /* out16 = 0 */                   \
  "eor    v25.16b,  v25.16b, v25.16b\n"  /* out17 = 0 */                   \
  "eor    v26.16b,  v26.16b, v26.16b\n"  /* out18 = 0 */                   \
  "eor    v27.16b,  v27.16b, v27.16b\n"  /* out19 = 0 */                   \
  "eor    v28.16b,  v28.16b, v28.16b\n"  /* out20 = 0 */                   \
  "eor    v29.16b,  v29.16b, v29.16b\n"  /* out21 = 0 */                   \
  "eor    v30.16b,  v30.16b, v30.16b\n"  /* out22 = 0 */                   \
  "eor    v31.16b,  v31.16b, v31.16b\n"  /* out23 = 0 */                   \
  "cbz    %w[k], 2f\n" /* check loop count > 0 */                          \
  /* main loop, unrool 0*/                                                 \
  "1:\n"                                 /* main loop */                   \
  "sdot   v8.4s ,  v4.16b,  v0.4b[0]\n"  /* out0 = b0 * a00[0], b0 = q4 */ \
  "sdot   v11.4s ,  v4.16b,  v0.4b[1]\n" /* out1 = b0 * a00[1], b0 = q4 */ \
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b2, b0 to q6, q7       */ \
  "sdot   v14.4s,  v4.16b,  v0.4b[2]\n"  /* out2 = b0 * a00[2], b0 = q4 */ \
  "sdot   v17.4s,  v4.16b,  v0.4b[3]\n"  /* out3 = b0 * a00[3], b0 = q4 */ \
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4     */ \
  "sdot   v20.4s,  v4.16b,  v1.4b[0]\n"  /* out4 = b0 * a01[0], b0 = q4 */ \
  "sdot   v23.4s,  v4.16b,  v1.4b[1]\n"  /* out5 = b0 * a01[1], b0 = q4 */ \
  "sdot   v26.4s,  v4.16b,  v1.4b[2]\n"  /* out6 = b0 * a01[2], b0 = q4 */ \
  "sdot   v29.4s,  v4.16b,  v1.4b[3]\n"  /* out7 = b0 * a01[3], b0 = q4 */ \
  "sdot   v9.4s,  v5.16b,  v0.4b[0]\n"   /* out8 = b1 * a00[0], b1 = q5 */ \
  "sdot   v12.4s,  v5.16b,  v0.4b[1]\n"  /* out9 = b1 * a00[1], b1 = q5 */ \
  "sdot   v15.4s,  v5.16b,  v0.4b[2]\n"  /* out10 = b1 * a00[2], b1 = q5*/ \
  "sdot   v18.4s,  v5.16b,  v0.4b[3]\n"  /* out11 = b1 * a00[3], b1 = q5*/ \
  "sdot   v21.4s,  v5.16b,  v1.4b[0]\n"  /* out12 = b1 * a01[0], b1 = q5*/ \
  "sdot   v24.4s,  v5.16b,  v1.4b[1]\n"  /* out13 = b1 * a01[1], b1 = q5*/ \
  "sdot   v27.4s,  v5.16b,  v1.4b[2]\n"  /* out14 = b1 * a01[2], b1 = q5*/ \
  "sdot   v30.4s,  v5.16b,  v1.4b[3]\n"  /* out15 = b1 * a01[3], b1 = q5*/ \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b1, b2 to q4, q5       */ \
  "sdot   v10.4s,  v6.16b,  v0.4b[0]\n"  /* out16 = b2 * a00[0], b2 = q6*/ \
  "sdot   v13.4s,  v6.16b,  v0.4b[1]\n"  /* out17 = b2 * a00[1], b2 = q6*/ \
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                                   \
  "sdot   v16.4s,  v6.16b,  v0.4b[2]\n" /* out18 = b2 * a00[2], b2 = q6*/  \
  "sdot   v19.4s,  v6.16b,  v0.4b[3]\n" /* out19 = b2 * a00[3], b2 = q6*/  \
  "sdot   v22.4s,  v6.16b,  v1.4b[0]\n" /* out20 = b2 * a00[0], b2 = q6*/  \
  "sdot   v25.4s,  v6.16b,  v1.4b[1]\n" /* out21 = b2 * a00[1], b2 = q6*/  \
  "sdot   v28.4s,  v6.16b,  v1.4b[2]\n" /* out22 = b2 * a00[2], b2 = q6*/  \
  "sdot   v31.4s,  v6.16b,  v1.4b[3]\n" /* out23 = b2 * a00[3], b2 = q6*/  \
  "ldp    q0, q1, [%[a_ptr]], #32\n"    /* load a00, a01 to q0, q1 */      \
  /* unrool 1 */                                                           \
  "sdot   v8.4s ,  v7.16b,  v2.4b[0]\n" /* out0 = b0 * a10[0], b0 = q7 */  \
  "sdot   v11.4s ,  v7.16b,  v2.4b[1]\n"/* out1 = b0 * a10[1], b0 = q7 */  \
  "sdot   v14.4s,  v7.16b,  v2.4b[2]\n" /* out2 = b0 * a10[2], b0 = q7 */  \
  "prfm   pldl1keep, [%[a_ptr], #256]\n"                                   \
  "sdot   v17.4s,  v7.16b,  v2.4b[3]\n" /* out3 = b0 * a10[3], b0 = q7 */  \
  "sdot   v20.4s,  v7.16b,  v3.4b[0]\n" /* out4 = b0 * a11[0], b0 = q7 */  \
  "sdot   v23.4s,  v7.16b,  v3.4b[1]\n" /* out5 = b0 * a11[1], b0 = q7 */  \
  "sdot   v26.4s,  v7.16b,  v3.4b[2]\n" /* out6 = b0 * a11[2], b0 = q7 */  \
  "sdot   v29.4s,  v7.16b,  v3.4b[3]\n" /* out7 = b0 * a11[3], b0 = q7 */  \
  "ldp    q6, q7, [%[b_ptr]], #32\n"    /* load b0, b1 to q6, q7       */  \
  "sdot   v9.4s,  v4.16b,  v2.4b[0]\n"  /* out8 = b0 * a10[0], b1 = q4 */  \
  "sdot   v12.4s,  v4.16b,  v2.4b[1]\n" /* out9 = b0 * a10[1], b1 = q4 */  \
  "sdot   v15.4s,  v4.16b,  v2.4b[2]\n" /* out10 = b1 * a10[2], b1 = q4*/  \
  "sdot   v18.4s,  v4.16b,  v2.4b[3]\n" /* out11 = b1 * a10[3], b1 = q4*/  \
  "sdot   v21.4s,  v4.16b,  v3.4b[0]\n" /* out12 = b1 * a10[0], b1 = q4*/  \
  "sdot   v24.4s,  v4.16b,  v3.4b[1]\n" /* out13 = b1 * a10[1], b1 = q4*/  \
  "sdot   v27.4s,  v4.16b,  v3.4b[2]\n" /* out14 = b1 * a10[2], b1 = q4*/  \
  "sdot   v30.4s,  v4.16b,  v3.4b[3]\n" /* out15 = b1 * a10[3], b1 = q4*/  \
  "sdot   v10.4s,  v5.16b,  v2.4b[0]\n" /* out16 = b2 * a10[0], b2 = q5*/  \
  "sdot   v13.4s,  v5.16b,  v2.4b[1]\n" /* out17 = b2 * a10[0], b2 = q5*/  \
  "sdot   v16.4s,  v5.16b,  v2.4b[2]\n" /* out18 = b2 * a10[0], b2 = q5*/  \
  "sdot   v19.4s,  v5.16b,  v2.4b[3]\n" /* out19 = b2 * a10[0], b2 = q5*/  \
  "sdot   v22.4s,  v5.16b,  v3.4b[0]\n" /* out20 = b2 * a10[0], b2 = q5*/  \
  "sdot   v25.4s,  v5.16b,  v3.4b[1]\n" /* out21 = b2 * a10[0], b2 = q5*/  \
  "sdot   v28.4s,  v5.16b,  v3.4b[2]\n" /* out22 = b2 * a10[0], b2 = q5*/  \
  "sdot   v31.4s,  v5.16b,  v3.4b[3]\n" /* out23 = b2 * a10[0], b2 = q5*/  \
  "ldp    q4, q5, [%[b_ptr]], #32\n"    /* load b2, b0 to q4, q5 */        \
  /* unrool 2*/                                                            \
  "sdot   v8.4s ,  v6.16b,  v0.4b[0]\n"  /* out0 = b0 * a00[0], b0 = q6 */ \
  "sdot   v11.4s ,  v6.16b,  v0.4b[1]\n" /* out1 = b0 * a00[1], b0 = q6 */ \
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4*/      \
  "sdot   v14.4s,  v6.16b,  v0.4b[2]\n"  /* out2 = b0 * a00[2], b0 = q6*/  \
  "sdot   v17.4s,  v6.16b,  v0.4b[3]\n"  /* out3 = b0 * a00[3], b0 = q6*/  \
  "sdot   v20.4s,  v6.16b,  v1.4b[0]\n"  /* out4 = b0 * a01[0], b0 = q6*/  \
  "sdot   v23.4s,  v6.16b,  v1.4b[1]\n"  /* out5 = b0 * a01[1], b0 = q6*/  \
  "sdot   v26.4s,  v6.16b,  v1.4b[2]\n"  /* out6 = b0 * a01[2], b0 = q6*/  \
  "sdot   v29.4s,  v6.16b,  v1.4b[3]\n"  /* out7 = b0 * a01[3], b0 = q6*/  \
  "sdot   v9.4s,  v7.16b,  v0.4b[0]\n"   /* out8 = b1 * a00[0], b1 = q7*/  \
  "sdot   v12.4s,  v7.16b,  v0.4b[1]\n"  /* out9 = b1 * a00[1], b1 = q7*/  \
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                                   \
  "sdot   v15.4s,  v7.16b,  v0.4b[2]\n" /* out10 = b1 * a00[2], b1 = q7*/  \
  "sdot   v18.4s,  v7.16b,  v0.4b[3]\n" /* out11 = b1 * a00[3], b1 = q7*/  \
  "sdot   v21.4s,  v7.16b,  v1.4b[0]\n" /* out12 = b1 * a01[0], b1 = q7*/  \
  "sdot   v24.4s,  v7.16b,  v1.4b[1]\n" /* out13 = b1 * a01[1], b1 = q7*/  \
  "sdot   v27.4s,  v7.16b,  v1.4b[2]\n" /* out14 = b1 * a01[2], b1 = q7*/  \
  "sdot   v30.4s,  v7.16b,  v1.4b[3]\n" /* out15 = b1 * a01[3], b1 = q7*/  \
  "ldp    q6, q7, [%[b_ptr]], #32\n"    /* load b1, b2 to q6, q7*/         \
  "sdot   v10.4s,  v4.16b,  v0.4b[0]\n" /* out16 = b2 * a00[0], b2 = q4*/  \
  "sdot   v13.4s,  v4.16b,  v0.4b[1]\n" /* out17 = b2 * a00[1], b2 = q4*/  \
  "sdot   v16.4s,  v4.16b,  v0.4b[2]\n" /* out18 = b2 * a00[2], b2 = q4*/  \
  "sdot   v19.4s,  v4.16b,  v0.4b[3]\n" /* out19 = b2 * a00[3], b2 = q4*/  \
  "sdot   v22.4s,  v4.16b,  v1.4b[0]\n" /* out20 = b2 * a00[0], b2 = q4*/  \
  "sdot   v25.4s,  v4.16b,  v1.4b[1]\n" /* out21 = b2 * a00[1], b2 = q4*/  \
  "sdot   v28.4s,  v4.16b,  v1.4b[2]\n" /* out22 = b2 * a00[2], b2 = q4*/  \
  "sdot   v31.4s,  v4.16b,  v1.4b[3]\n" /* out23 = b2 * a00[3], b2 = q4*/  \
  "ldp    q0, q1, [%[a_ptr]], #32\n" /* load a00, a01 to q0, q1*/          \
  /* unrool 3*/                                                            \
  "sdot   v8.4s ,  v5.16b,  v2.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v5.16b,  v2.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v5.16b,  v2.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v5.16b,  v2.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v5.16b,  v3.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v5.16b,  v3.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v5.16b,  v3.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v5.16b,  v3.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b0, b1 to q4, q5*/        \
  "sdot   v9.4s,  v6.16b,  v2.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v6.16b,  v2.4b[1]\n"  /* out9 = b0 * a10[1], b1 = q6*/  \
  "prfm   pldl1keep, [%[a_ptr], #256]\n"                                   \
  "sdot   v15.4s,  v6.16b,  v2.4b[2]\n" /* out10 = b1 * a10[2], b1 = q6*/  \
  "sdot   v18.4s,  v6.16b,  v2.4b[3]\n" /* out11 = b1 * a10[3], b1 = q6*/  \
  "sdot   v21.4s,  v6.16b,  v3.4b[0]\n" /* out12 = b1 * a10[0], b1 = q6*/  \
  "sdot   v24.4s,  v6.16b,  v3.4b[1]\n" /* out13 = b1 * a10[1], b1 = q6*/  \
  "sdot   v27.4s,  v6.16b,  v3.4b[2]\n" /* out14 = b1 * a10[2], b1 = q6*/  \
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                                   \
  "sdot   v30.4s,  v6.16b,  v3.4b[3]\n" /* out15 = b1 * a10[3], b1 = q6*/  \
  "sdot   v10.4s,  v7.16b,  v2.4b[0]\n" /* out16 = b2 * a10[0], b2 = q7*/  \
  "sdot   v13.4s,  v7.16b,  v2.4b[1]\n" /* out17 = b2 * a10[0], b2 = q7*/  \
  "sdot   v16.4s,  v7.16b,  v2.4b[2]\n" /* out18 = b2 * a10[0], b2 = q7*/  \
  "sdot   v19.4s,  v7.16b,  v2.4b[3]\n" /* out19 = b2 * a10[0], b2 = q7*/  \
  "sdot   v22.4s,  v7.16b,  v3.4b[0]\n" /* out20 = b2 * a10[0], b2 = q7*/  \
  "sdot   v25.4s,  v7.16b,  v3.4b[1]\n" /* out21 = b2 * a10[0], b2 = q7*/  \
  "subs   %w[k], %w[k], #1\n"           /* loop count - 1*/                \
  "sdot   v28.4s,  v7.16b,  v3.4b[2]\n" /* out22 = b2 * a10[0], b2 = q7*/  \
  "sdot   v31.4s,  v7.16b,  v3.4b[3]\n" /* out23 = b2 * a10[0], b2 = q7*/  \
  "bne    1b\n" /* Target to use when K is 1 or 2 */                       \
  "2:\n"                                             /* process tail*/     \
  "subs       %w[tail], %w[tail], #1\n"              /* tail--*/           \
  "beq        3f\n" /*jump to tail = 1*/                                   \
  /* final unrool 0, unrool 0, tail > 1*/                                  \
  "sdot   v8.4s ,  v4.16b,  v0.4b[0]\n"  /* out0 = b0 * a00[0], b0 = q4*/  \
  "sdot   v11.4s ,  v4.16b,  v0.4b[1]\n" /* out1 = b0 * a00[1], b0 = q4*/  \
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b2, b0 to q6, q7*/        \
  "sdot   v14.4s,  v4.16b,  v0.4b[2]\n"  /* out2 = b0 * a00[2], b0 = q4*/  \
  "sdot   v17.4s,  v4.16b,  v0.4b[3]\n"  /* out3 = b0 * a00[3], b0 = q4*/  \
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q2, q3*/      \
  "sdot   v20.4s,  v4.16b,  v1.4b[0]\n"  /* out4 = b0 * a01[0], b0 = q4*/  \
  "sdot   v23.4s,  v4.16b,  v1.4b[1]\n"  /* out5 = b0 * a01[1], b0 = q4*/  \
  "sdot   v26.4s,  v4.16b,  v1.4b[2]\n"  /* out6 = b0 * a01[2], b0 = q4*/  \
  "sdot   v29.4s,  v4.16b,  v1.4b[3]\n"  /* out7 = b0 * a01[3], b0 = q4*/  \
  "subs   %w[tail], %w[tail], #1\n"      /* tail--*/                       \
  "sdot   v9.4s,  v5.16b,  v0.4b[0]\n"   /* out8 = b1 * a00[0], b1 = q5*/  \
  "sdot   v12.4s,  v5.16b,  v0.4b[1]\n"  /* out9 = b1 * a00[1], b1 = q5*/  \
  "sdot   v15.4s,  v5.16b,  v0.4b[2]\n"  /* out10 = b1 * a00[2], b1 = q5*/ \
  "sdot   v18.4s,  v5.16b,  v0.4b[3]\n"  /* out11 = b1 * a00[3], b1 = q5*/ \
  "sdot   v21.4s,  v5.16b,  v1.4b[0]\n"  /* out12 = b1 * a01[0], b1 = q5*/ \
  "sdot   v24.4s,  v5.16b,  v1.4b[1]\n"  /* out13 = b1 * a01[1], b1 = q5*/ \
  "sdot   v27.4s,  v5.16b,  v1.4b[2]\n"  /* out14 = b1 * a01[2], b1 = q5*/ \
  "sdot   v30.4s,  v5.16b,  v1.4b[3]\n"  /* out15 = b1 * a01[3], b1 = q5*/ \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b1, b2 to q4, q5*/        \
  "sdot   v10.4s,  v6.16b,  v0.4b[0]\n"  /* out16 = b2 * a00[0], b2 = q6*/ \
  "sdot   v13.4s,  v6.16b,  v0.4b[1]\n"  /* out17 = b2 * a00[1], b2 = q6*/ \
  "sdot   v16.4s,  v6.16b,  v0.4b[2]\n"  /* out18 = b2 * a00[2], b2 = q6*/ \
  "sdot   v19.4s,  v6.16b,  v0.4b[3]\n"  /* out19 = b2 * a00[3], b2 = q6*/ \
  "sdot   v22.4s,  v6.16b,  v1.4b[0]\n"  /* out20 = b2 * a00[0], b2 = q6*/ \
  "sdot   v25.4s,  v6.16b,  v1.4b[1]\n"  /* out21 = b2 * a00[1], b2 = q6*/ \
  "sdot   v28.4s,  v6.16b,  v1.4b[2]\n"  /* out22 = b2 * a00[2], b2 = q6*/ \
  "sdot   v31.4s,  v6.16b,  v1.4b[3]\n"  /* out23 = b2 * a00[3], b2 = q6*/ \
  "beq        4f\n" /*jump to tail = 2*/                                   \
  /* unrool 1, tail > 2*/                                                  \
  "ldp    q0, q1, [%[a_ptr]], #32\n"     /* load a00, a01 to q0, q1*/      \
  "sdot   v8.4s ,  v7.16b,  v2.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q7*/  \
  "sdot   v11.4s ,  v7.16b,  v2.4b[1]\n" /* out1 = b0 * a10[1], b0 = q7*/  \
  "sdot   v14.4s,  v7.16b,  v2.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q7*/  \
  "sdot   v17.4s,  v7.16b,  v2.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q7*/  \
  "sdot   v20.4s,  v7.16b,  v3.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q7*/  \
  "sdot   v23.4s,  v7.16b,  v3.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q7*/  \
  "sdot   v26.4s,  v7.16b,  v3.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q7*/  \
  "sdot   v29.4s,  v7.16b,  v3.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q7*/  \
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b0, b1 to q6, q7*/        \
  "sdot   v9.4s,  v4.16b,  v2.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q4*/  \
  "sdot   v12.4s,  v4.16b,  v2.4b[1]\n"  /* out9 = b0 * a10[1], b1 = q4*/  \
  "sdot   v15.4s,  v4.16b,  v2.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q4*/ \
  "sdot   v18.4s,  v4.16b,  v2.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q4*/ \
  "sdot   v21.4s,  v4.16b,  v3.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q4*/ \
  "sdot   v24.4s,  v4.16b,  v3.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q4*/ \
  "sdot   v27.4s,  v4.16b,  v3.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q4*/ \
  "sdot   v30.4s,  v4.16b,  v3.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q4*/ \
  "subs   %w[tail], %w[tail], #1\n"      /* tail--*/                       \
  "sdot   v10.4s,  v5.16b,  v2.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q5*/ \
  "sdot   v13.4s,  v5.16b,  v2.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q5*/ \
  "sdot   v16.4s,  v5.16b,  v2.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q5*/ \
  "sdot   v19.4s,  v5.16b,  v2.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q5*/ \
  "sdot   v22.4s,  v5.16b,  v3.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q5*/ \
  "sdot   v25.4s,  v5.16b,  v3.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q5*/ \
  "sdot   v28.4s,  v5.16b,  v3.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q5*/ \
  "sdot   v31.4s,  v5.16b,  v3.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q5*/ \
  "beq        5f\n" /*jump to tail = 3*/                                   \
  /* unrool 2, tail = 4*/                                                  \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b2, b0 to q4, q5*/        \
  "sdot   v8.4s ,  v6.16b,  v0.4b[0]\n"  /* out0 = b0 * a00[0], b0 = q6*/  \
  "sdot   v11.4s ,  v6.16b,  v0.4b[1]\n" /* out1 = b0 * a00[1], b0 = q6*/  \
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4*/      \
  "sdot   v14.4s,  v6.16b,  v0.4b[2]\n"  /* out2 = b0 * a00[2], b0 = q6*/  \
  "sdot   v17.4s,  v6.16b,  v0.4b[3]\n"  /* out3 = b0 * a00[3], b0 = q6*/  \
  "sdot   v20.4s,  v6.16b,  v1.4b[0]\n"  /* out4 = b0 * a01[0], b0 = q6*/  \
  "sdot   v23.4s,  v6.16b,  v1.4b[1]\n"  /* out5 = b0 * a01[1], b0 = q6*/  \
  "sdot   v26.4s,  v6.16b,  v1.4b[2]\n"  /* out6 = b0 * a01[2], b0 = q6*/  \
  "sdot   v29.4s,  v6.16b,  v1.4b[3]\n"  /* out7 = b0 * a01[3], b0 = q6*/  \
  "sdot   v9.4s,  v7.16b,  v0.4b[0]\n"   /* out8 = b1 * a00[0], b1 = q7*/  \
  "sdot   v12.4s,  v7.16b,  v0.4b[1]\n"  /* out9 = b1 * a00[1], b1 = q7*/  \
  "sdot   v15.4s,  v7.16b,  v0.4b[2]\n"  /* out10 = b1 * a00[2], b1 = q7*/ \
  "sdot   v18.4s,  v7.16b,  v0.4b[3]\n"  /* out11 = b1 * a00[3], b1 = q7*/ \
  "sdot   v21.4s,  v7.16b,  v1.4b[0]\n"  /* out12 = b1 * a01[0], b1 = q7*/ \
  "sdot   v24.4s,  v7.16b,  v1.4b[1]\n"  /* out13 = b1 * a01[1], b1 = q7*/ \
  "sdot   v27.4s,  v7.16b,  v1.4b[2]\n"  /* out14 = b1 * a01[2], b1 = q7*/ \
  "sdot   v30.4s,  v7.16b,  v1.4b[3]\n"  /* out15 = b1 * a01[3], b1 = q7*/ \
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b1, b2 to q6, q7*/        \
  "sdot   v10.4s,  v4.16b,  v0.4b[0]\n"  /* out16 = b2 * a00[0], b2 = q4*/ \
  "sdot   v13.4s,  v4.16b,  v0.4b[1]\n"  /* out17 = b2 * a00[1], b2 = q4*/ \
  "sdot   v16.4s,  v4.16b,  v0.4b[2]\n"  /* out18 = b2 * a00[2], b2 = q4*/ \
  "sdot   v19.4s,  v4.16b,  v0.4b[3]\n"  /* out19 = b2 * a00[3], b2 = q4*/ \
  "sdot   v22.4s,  v4.16b,  v1.4b[0]\n"  /* out20 = b2 * a00[0], b2 = q4*/ \
  "sdot   v25.4s,  v4.16b,  v1.4b[1]\n"  /* out21 = b2 * a00[1], b2 = q4*/ \
  "sdot   v28.4s,  v4.16b,  v1.4b[2]\n"  /* out22 = b2 * a00[2], b2 = q4*/ \
  "sdot   v31.4s,  v4.16b,  v1.4b[3]\n"  /* out23 = b2 * a00[3], b2 = q4*/ \
  /* unrool 3, tail = 4*/                                                  \
  "sdot   v8.4s ,  v5.16b,  v2.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v5.16b,  v2.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v5.16b,  v2.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v5.16b,  v2.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v5.16b,  v3.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v5.16b,  v3.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v5.16b,  v3.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v5.16b,  v3.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "sdot   v9.4s,  v6.16b,  v2.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v6.16b,  v2.4b[1]\n"  /* out9 = b1 * a10[1], b1 = q6*/  \
  "sdot   v15.4s,  v6.16b,  v2.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q6*/ \
  "sdot   v18.4s,  v6.16b,  v2.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q6*/ \
  "sdot   v21.4s,  v6.16b,  v3.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q6*/ \
  "sdot   v24.4s,  v6.16b,  v3.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q6*/ \
  "sdot   v27.4s,  v6.16b,  v3.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q6*/ \
  "sdot   v30.4s,  v6.16b,  v3.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q6*/ \
  "sdot   v10.4s,  v7.16b,  v2.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q7*/ \
  "sdot   v13.4s,  v7.16b,  v2.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q7*/ \
  "sdot   v16.4s,  v7.16b,  v2.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q7*/ \
  "sdot   v19.4s,  v7.16b,  v2.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q7*/ \
  "sdot   v22.4s,  v7.16b,  v3.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q7*/ \
  "sdot   v25.4s,  v7.16b,  v3.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q7*/ \
  "sdot   v28.4s,  v7.16b,  v3.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q7*/ \
  "sdot   v31.4s,  v7.16b,  v3.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q7*/ \
  "b      11f\n"                         /* tails==1 final tail*/          \
  "3: \n"                                /* tail=1*/                       \
  "ldr    q6, [%[b_ptr]], #16\n"         /* load b2 to q6*/                \
  "sdot   v8.4s ,  v4.16b,  v0.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v4.16b,  v0.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v4.16b,  v0.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v4.16b,  v0.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v4.16b,  v1.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v4.16b,  v1.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v4.16b,  v1.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v4.16b,  v1.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "sdot   v9.4s,  v5.16b,  v0.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v5.16b,  v0.4b[1]\n"  /* out9 = b1 * a10[1], b1 = q6*/  \
  "sdot   v15.4s,  v5.16b,  v0.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q6*/ \
  "sdot   v18.4s,  v5.16b,  v0.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q6*/ \
  "sdot   v21.4s,  v5.16b,  v1.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q6*/ \
  "sdot   v24.4s,  v5.16b,  v1.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q6*/ \
  "sdot   v27.4s,  v5.16b,  v1.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q6*/ \
  "sdot   v30.4s,  v5.16b,  v1.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q6*/ \
  "sdot   v10.4s,  v6.16b,  v0.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q7*/ \
  "sdot   v13.4s,  v6.16b,  v0.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q7*/ \
  "sdot   v16.4s,  v6.16b,  v0.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q7*/ \
  "sdot   v19.4s,  v6.16b,  v0.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q7*/ \
  "sdot   v22.4s,  v6.16b,  v1.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q7*/ \
  "sdot   v25.4s,  v6.16b,  v1.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q7*/ \
  "sdot   v28.4s,  v6.16b,  v1.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q7*/ \
  "sdot   v31.4s,  v6.16b,  v1.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q7*/ \
  "b      11f\n"                         /* tails==2 final tail*/          \
  "4:\n"                                 /* tail = 2*/                     \
  "sdot   v8.4s ,  v7.16b,  v2.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v7.16b,  v2.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v7.16b,  v2.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v7.16b,  v2.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v7.16b,  v3.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v7.16b,  v3.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v7.16b,  v3.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v7.16b,  v3.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "sdot   v9.4s,  v4.16b,  v2.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v4.16b,  v2.4b[1]\n"  /* out9 = b1 * a10[1], b1 = q6*/  \
  "sdot   v15.4s,  v4.16b,  v2.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q6*/ \
  "sdot   v18.4s,  v4.16b,  v2.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q6*/ \
  "sdot   v21.4s,  v4.16b,  v3.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q6*/ \
  "sdot   v24.4s,  v4.16b,  v3.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q6*/ \
  "sdot   v27.4s,  v4.16b,  v3.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q6*/ \
  "sdot   v30.4s,  v4.16b,  v3.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q6*/ \
  "sdot   v10.4s,  v5.16b,  v2.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q7*/ \
  "sdot   v13.4s,  v5.16b,  v2.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q7*/ \
  "sdot   v16.4s,  v5.16b,  v2.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q7*/ \
  "sdot   v19.4s,  v5.16b,  v2.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q7*/ \
  "sdot   v22.4s,  v5.16b,  v3.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q7*/ \
  "sdot   v25.4s,  v5.16b,  v3.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q7*/ \
  "sdot   v28.4s,  v5.16b,  v3.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q7*/ \
  "sdot   v31.4s,  v5.16b,  v3.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q7*/ \
  "b      11f\n"                         /* tails==3 final tail*/          \
  "5:\n"                                 /* tail = 3*/                     \
  "ldr    q4, [%[b_ptr]], #16\n"         /* load b2, b0 to q4*/            \
  "sdot   v8.4s ,  v6.16b,  v0.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v6.16b,  v0.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v6.16b,  v0.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v6.16b,  v0.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v6.16b,  v1.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v6.16b,  v1.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v6.16b,  v1.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v6.16b,  v1.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "sdot   v9.4s,  v7.16b,  v0.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v7.16b,  v0.4b[1]\n"  /* out9 = b1 * a10[1], b1 = q6*/  \
  "sdot   v15.4s,  v7.16b,  v0.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q6*/ \
  "sdot   v18.4s,  v7.16b,  v0.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q6*/ \
  "sdot   v21.4s,  v7.16b,  v1.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q6*/ \
  "sdot   v24.4s,  v7.16b,  v1.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q6*/ \
  "sdot   v27.4s,  v7.16b,  v1.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q6*/ \
  "sdot   v30.4s,  v7.16b,  v1.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q6*/ \
  "sdot   v10.4s,  v4.16b,  v0.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q7*/ \
  "sdot   v13.4s,  v4.16b,  v0.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q7*/ \
  "sdot   v16.4s,  v4.16b,  v0.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q7*/ \
  "sdot   v19.4s,  v4.16b,  v0.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q7*/ \
  "sdot   v22.4s,  v4.16b,  v1.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q7*/ \
  "sdot   v25.4s,  v4.16b,  v1.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q7*/ \
  "sdot   v28.4s,  v4.16b,  v1.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q7*/ \
  "sdot   v31.4s,  v4.16b,  v1.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q7*/ \
  "11: \n"                               /* end */
#endif

#define GEMM_SDOT_RELU_8x4                         \
  "cmp    %w[relu],   #0\n"       /* skip relu */  \
  "beq    12f\n"                                   \
  "cmp    %w[relu],    #1\n"    /* skip relu */    \
  "movi   v2.4s, #0\n"             /* for relu*/   \
  "bne    13f\n"                /* other act */    \
  "fmax   v8.4s, v8.4s, v2.4s\n"   /* relu*/       \
  "fmax   v11.4s, v11.4s, v2.4s\n" /* relu*/       \
  "fmax   v14.4s, v14.4s, v2.4s\n" /* relu*/       \
  "fmax   v17.4s,v17.4s,v2.4s\n"   /* relu*/       \
  "fmax   v20.4s, v20.4s, v2.4s\n" /* relu*/       \
  "fmax   v23.4s, v23.4s, v2.4s\n" /* relu*/       \
  "fmax   v26.4s, v26.4s, v2.4s\n" /* relu*/       \
  "fmax   v29.4s, v29.4s, v2.4s\n" /* relu*/       \
  "b      12f                    \n"   /* relu end */

#define GEMM_SDOT_RELU_8x8                         \
  "cmp    %w[relu],   #0\n"       /* skip relu */  \
  "beq    12f\n"                                   \
  "cmp    %w[relu],    #1\n"    /* skip relu */    \
  "movi   v2.4s, #0\n"             /* for relu*/   \
  "bne    13f\n"                /* other act */    \
  "fmax   v8.4s, v8.4s, v2.4s\n"   /* relu*/       \
  "fmax   v9.4s, v9.4s, v2.4s\n"   /* relu*/       \
  "fmax   v11.4s, v11.4s, v2.4s\n" /* relu*/       \
  "fmax   v12.4s, v12.4s, v2.4s\n" /* relu*/       \
  "fmax   v14.4s, v14.4s, v2.4s\n" /* relu*/       \
  "fmax   v15.4s, v15.4s, v2.4s\n" /* relu*/       \
  "fmax   v17.4s,v17.4s,v2.4s\n"   /* relu*/       \
  "fmax   v18.4s, v18.4s, v2.4s\n" /* relu*/       \
  "fmax   v20.4s, v20.4s, v2.4s\n" /* relu*/       \
  "fmax   v21.4s, v21.4s, v2.4s\n" /* relu*/       \
  "fmax   v23.4s, v23.4s, v2.4s\n" /* relu*/       \
  "fmax   v24.4s, v24.4s, v2.4s\n" /* relu*/       \
  "fmax   v26.4s, v26.4s, v2.4s\n" /* relu*/       \
  "fmax   v27.4s, v27.4s, v2.4s\n" /* relu*/       \
  "fmax   v29.4s, v29.4s, v2.4s\n" /* relu*/       \
  "fmax   v30.4s, v30.4s, v2.4s\n" /* relu*/       \
  "b      12f                    \n"   /* relu end */

#define GEMM_SDOT_RELU                             \
  "cmp    %w[relu],   #0\n"       /* skip relu */  \
  "beq    12f\n"                                   \
  "cmp    %w[relu],    #1\n"    /* skip relu */    \
  "movi   v2.4s, #0\n"             /* for relu*/   \
  "bne    13f\n"                /* other act */    \
  "fmax   v8.4s, v8.4s, v2.4s\n"   /* relu*/       \
  "fmax   v9.4s, v9.4s, v2.4s\n"   /* relu*/       \
  "fmax   v10.4s, v10.4s, v2.4s\n" /* relu*/       \
  "fmax   v11.4s, v11.4s, v2.4s\n" /* relu*/       \
  "fmax   v12.4s, v12.4s, v2.4s\n" /* relu*/       \
  "fmax   v13.4s, v13.4s, v2.4s\n" /* relu*/       \
  "fmax   v14.4s, v14.4s, v2.4s\n" /* relu*/       \
  "fmax   v15.4s, v15.4s, v2.4s\n" /* relu*/       \
  "fmax   v16.4s,v16.4s,v2.4s\n"   /* relu*/       \
  "fmax   v17.4s,v17.4s,v2.4s\n"   /* relu*/       \
  "fmax   v18.4s, v18.4s, v2.4s\n" /* relu*/       \
  "fmax   v19.4s, v19.4s, v2.4s\n" /* relu*/       \
  "fmax   v20.4s, v20.4s, v2.4s\n" /* relu*/       \
  "fmax   v21.4s, v21.4s, v2.4s\n" /* relu*/       \
  "fmax   v22.4s, v22.4s, v2.4s\n" /* relu*/       \
  "fmax   v23.4s, v23.4s, v2.4s\n" /* relu*/       \
  "fmax   v24.4s, v24.4s, v2.4s\n" /* relu*/       \
  "fmax   v25.4s, v25.4s, v2.4s\n" /* relu*/       \
  "fmax   v26.4s, v26.4s, v2.4s\n" /* relu*/       \
  "fmax   v27.4s, v27.4s, v2.4s\n" /* relu*/       \
  "fmax   v28.4s, v28.4s, v2.4s\n" /* relu*/       \
  "fmax   v29.4s, v29.4s, v2.4s\n" /* relu*/       \
  "fmax   v30.4s, v30.4s, v2.4s\n" /* relu*/       \
  "fmax   v31.4s, v31.4s, v2.4s\n" /* relu*/       \
  "b      12f                    \n"   /* relu end */

#define GEMM_SDOT_RELU6_8x4                               \
  "13:    \n"                                             \
  "cmp    %w[relu],   #2\n"       /* skip relu6 */        \
  "bne   14f\n"                                           \
  "fmax   v8.4s, v8.4s, v2.4s\n"   /* relu*/              \
  "fmax   v11.4s, v11.4s, v2.4s\n" /* relu*/              \
  "ld1    {v3.4s}, [%[alpha]]    \n"    /* relu6 alpha */ \
  "fmax   v14.4s, v14.4s, v2.4s\n" /* relu*/              \
  "fmax   v17.4s,v17.4s,v2.4s\n"   /* relu*/              \
  "fmax   v20.4s, v20.4s, v2.4s\n" /* relu*/              \
  "fmax   v23.4s, v23.4s, v2.4s\n" /* relu*/              \
  "fmax   v26.4s, v26.4s, v2.4s\n" /* relu*/              \
  "fmax   v29.4s, v29.4s, v2.4s\n" /* relu*/              \
  "fmin   v8.4s, v8.4s, v3.4s\n"   /* relu6*/             \
  "fmin   v11.4s, v11.4s, v3.4s\n" /* relu6*/             \
  "fmin   v14.4s, v14.4s, v3.4s\n" /* relu6*/             \
  "fmin   v17.4s, v17.4s, v3.4s\n"   /* relu6*/           \
  "fmin   v20.4s, v20.4s, v3.4s\n" /* relu6*/             \
  "fmin   v23.4s, v23.4s, v3.4s\n" /* relu6*/             \
  "fmin   v26.4s, v26.4s, v3.4s\n" /* relu6*/             \
  "fmin   v29.4s, v29.4s, v3.4s\n" /* relu6*/             \
  "b      12f                    \n"   /* relu end */

#define GEMM_SDOT_RELU6_8x8                               \
  "13:    \n"                                             \
  "cmp    %w[relu],   #2\n"       /* skip relu6 */        \
  "bne   14f\n"                                           \
  "fmax   v8.4s, v8.4s, v2.4s\n"   /* relu*/              \
  "fmax   v9.4s, v9.4s, v2.4s\n"   /* relu*/              \
  "fmax   v11.4s, v11.4s, v2.4s\n" /* relu*/              \
  "ld1    {v3.4s}, [%[alpha]]    \n"    /* relu6 alpha */ \
  "fmax   v12.4s, v12.4s, v2.4s\n" /* relu*/              \
  "fmax   v14.4s, v14.4s, v2.4s\n" /* relu*/              \
  "fmax   v15.4s, v15.4s, v2.4s\n" /* relu*/              \
  "fmax   v17.4s,v17.4s,v2.4s\n"   /* relu*/              \
  "fmax   v18.4s, v18.4s, v2.4s\n" /* relu*/              \
  "fmax   v20.4s, v20.4s, v2.4s\n" /* relu*/              \
  "fmax   v21.4s, v21.4s, v2.4s\n" /* relu*/              \
  "fmax   v23.4s, v23.4s, v2.4s\n" /* relu*/              \
  "fmax   v24.4s, v24.4s, v2.4s\n" /* relu*/              \
  "fmax   v26.4s, v26.4s, v2.4s\n" /* relu*/              \
  "fmax   v27.4s, v27.4s, v2.4s\n" /* relu*/              \
  "fmax   v29.4s, v29.4s, v2.4s\n" /* relu*/              \
  "fmax   v30.4s, v30.4s, v2.4s\n" /* relu*/              \
  "fmin   v8.4s, v8.4s, v3.4s\n"   /* relu6*/             \
  "fmin   v9.4s, v9.4s, v3.4s\n"   /* relu6*/             \
  "fmin   v11.4s, v11.4s, v3.4s\n" /* relu6*/             \
  "fmin   v12.4s, v12.4s, v3.4s\n" /* relu6*/             \
  "fmin   v14.4s, v14.4s, v3.4s\n" /* relu6*/             \
  "fmin   v15.4s, v15.4s, v3.4s\n" /* relu6*/             \
  "fmin   v17.4s, v17.4s, v3.4s\n"   /* relu6*/           \
  "fmin   v18.4s, v18.4s, v3.4s\n" /* relu6*/             \
  "fmin   v20.4s, v20.4s, v3.4s\n" /* relu6*/             \
  "fmin   v21.4s, v21.4s, v3.4s\n" /* relu6*/             \
  "fmin   v23.4s, v23.4s, v3.4s\n" /* relu6*/             \
  "fmin   v24.4s, v24.4s, v3.4s\n" /* relu6*/             \
  "fmin   v26.4s, v26.4s, v3.4s\n" /* relu6*/             \
  "fmin   v27.4s, v27.4s, v3.4s\n" /* relu6*/             \
  "fmin   v29.4s, v29.4s, v3.4s\n" /* relu6*/             \
  "fmin   v30.4s, v30.4s, v3.4s\n" /* relu6*/             \
  "b      12f                    \n"   /* relu end */

#define GEMM_SDOT_RELU6                            \
  "13:    \n"                                      \
  "cmp    %w[relu],   #2\n"       /* skip relu6 */ \
  "bne   14f\n"                                    \
  "movi   v2.4s, #0\n"             /* for relu*/   \
  "fmax   v8.4s, v8.4s, v2.4s\n"   /* relu*/       \
  "fmax   v9.4s, v9.4s, v2.4s\n"   /* relu*/       \
  "fmax   v10.4s, v10.4s, v2.4s\n" /* relu*/       \
  "fmax   v11.4s, v11.4s, v2.4s\n" /* relu*/       \
  "ld1    {v3.4s}, [%[alpha]]    \n"    /* relu6 alpha */ \
  "fmax   v12.4s, v12.4s, v2.4s\n" /* relu*/       \
  "fmax   v13.4s, v13.4s, v2.4s\n" /* relu*/       \
  "fmax   v14.4s, v14.4s, v2.4s\n" /* relu*/       \
  "fmax   v15.4s, v15.4s, v2.4s\n" /* relu*/       \
  "fmax   v16.4s,v16.4s,v2.4s\n"   /* relu*/       \
  "fmax   v17.4s,v17.4s,v2.4s\n"   /* relu*/       \
  "fmax   v18.4s, v18.4s, v2.4s\n" /* relu*/       \
  "fmax   v19.4s, v19.4s, v2.4s\n" /* relu*/       \
  "fmax   v20.4s, v20.4s, v2.4s\n" /* relu*/       \
  "fmax   v21.4s, v21.4s, v2.4s\n" /* relu*/       \
  "fmax   v22.4s, v22.4s, v2.4s\n" /* relu*/       \
  "fmax   v23.4s, v23.4s, v2.4s\n" /* relu*/       \
  "fmax   v24.4s, v24.4s, v2.4s\n" /* relu*/       \
  "fmax   v25.4s, v25.4s, v2.4s\n" /* relu*/       \
  "fmax   v26.4s, v26.4s, v2.4s\n" /* relu*/       \
  "fmax   v27.4s, v27.4s, v2.4s\n" /* relu*/       \
  "fmax   v28.4s, v28.4s, v2.4s\n" /* relu*/       \
  "fmax   v29.4s, v29.4s, v2.4s\n" /* relu*/       \
  "fmax   v30.4s, v30.4s, v2.4s\n" /* relu*/       \
  "fmax   v31.4s, v31.4s, v2.4s\n" /* relu*/       \
  "fmin   v8.4s, v8.4s, v3.4s\n"   /* relu6*/       \
  "fmin   v9.4s, v9.4s, v3.4s\n"   /* relu6*/       \
  "fmin   v10.4s, v10.4s, v3.4s\n" /* relu6*/       \
  "fmin   v11.4s, v11.4s, v3.4s\n" /* relu6*/       \
  "fmin   v12.4s, v12.4s, v3.4s\n" /* relu6*/       \
  "fmin   v13.4s, v13.4s, v3.4s\n" /* relu6*/       \
  "fmin   v14.4s, v14.4s, v3.4s\n" /* relu6*/       \
  "fmin   v15.4s, v15.4s, v3.4s\n" /* relu6*/       \
  "fmin   v16.4s, v16.4s, v3.4s\n"   /* relu6*/     \
  "fmin   v17.4s, v17.4s, v3.4s\n"   /* relu6*/     \
  "fmin   v18.4s, v18.4s, v3.4s\n" /* relu6*/       \
  "fmin   v19.4s, v19.4s, v3.4s\n" /* relu6*/       \
  "fmin   v20.4s, v20.4s, v3.4s\n" /* relu6*/       \
  "fmin   v21.4s, v21.4s, v3.4s\n" /* relu6*/       \
  "fmin   v22.4s, v22.4s, v3.4s\n" /* relu6*/       \
  "fmin   v23.4s, v23.4s, v3.4s\n" /* relu6*/       \
  "fmin   v24.4s, v24.4s, v3.4s\n" /* relu6*/       \
  "fmin   v25.4s, v25.4s, v3.4s\n" /* relu6*/       \
  "fmin   v26.4s, v26.4s, v3.4s\n" /* relu6*/       \
  "fmin   v27.4s, v27.4s, v3.4s\n" /* relu6*/       \
  "fmin   v28.4s, v28.4s, v3.4s\n" /* relu6*/       \
  "fmin   v29.4s, v29.4s, v3.4s\n" /* relu6*/       \
  "fmin   v30.4s, v30.4s, v3.4s\n" /* relu6*/       \
  "fmin   v31.4s, v31.4s, v3.4s\n" /* relu6*/       \
  "b      12f                    \n"   /* relu end */

#define GEMM_SDOT_LEAKY_RELU_8x4                            \
  "14: \n"                                                  \
  "cmp    %w[relu],   #3\n"       /* skip relu6 */          \
  "bne   15f\n"                                             \
  "ld1    {v3.4s}, [%[alpha]]\n"   /* leakyrelu alpha */    \
  "fcmge  v4.4s,    v8.4s,    v2.4s   \n" /* vcgeq_f32 */   \
  "fmul   v5.4s,    v8.4s,    v3.4s   \n" /* vmulq_f32 */   \
  "bif    v8.16b,   v5.16b,   v4.16b  \n" /* choose*/       \
  "fcmge  v6.4s,    v11.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v11.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v11.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v14.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v14.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v14.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "fcmge  v6.4s,    v17.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v17.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v17.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v20.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v20.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v20.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "fcmge  v6.4s,    v23.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v23.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v23.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v26.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v26.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v26.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "fcmge  v6.4s,    v29.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v29.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v29.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "b 12f \n"

#define GEMM_SDOT_LEAKY_RELU_8x8                            \
  "14: \n"                                                  \
  "cmp    %w[relu],   #3\n"       /* skip relu6 */          \
  "bne   15f\n"                                             \
  "ld1    {v3.4s}, [%[alpha]]\n"   /* leakyrelu alpha */    \
  "fcmge  v4.4s,    v8.4s,    v2.4s   \n" /* vcgeq_f32 */   \
  "fmul   v5.4s,    v8.4s,    v3.4s   \n" /* vmulq_f32 */   \
  "fcmge  v6.4s,    v9.4s,    v2.4s   \n" /* vcgeq_f32 */   \
  "fmul   v7.4s,    v9.4s,    v3.4s   \n" /* vmulq_f32 */   \
  "bif    v8.16b,   v5.16b,   v4.16b  \n" /* choose*/       \
  "bif    v9.16b,   v7.16b,   v6.16b  \n" /* choose*/       \
  "fcmge  v6.4s,    v11.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v11.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v11.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v12.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v12.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v12.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v14.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v14.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v15.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v15.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v14.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v15.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v6.4s,    v17.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v17.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v17.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v18.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v18.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v18.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v20.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v20.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v21.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v21.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v20.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v21.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v6.4s,    v23.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v23.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v23.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v24.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v24.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v24.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v26.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v26.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v27.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v27.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v26.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v27.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v6.4s,    v29.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v29.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v29.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v30.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v30.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v30.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "b 12f \n"

#define GEMM_SDOT_LEAKY_RELU                        \
  "14: \n"                                           \
  "cmp    %w[relu],   #3\n"       /* skip relu6 */          \
  "bne   15f\n"                                             \
  "ld1    {v3.4s}, [%[alpha]]\n"   /* leakyrelu alpha */    \
  "fcmge  v4.4s,    v8.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v8.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v9.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v9.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v8.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v9.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v10.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v10.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v11.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v11.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v10.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v11.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v12.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v12.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v13.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v13.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v12.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v13.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v14.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v14.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v15.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v15.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v14.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v15.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v16.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v16.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v17.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v17.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v16.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v17.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v18.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v18.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v19.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v19.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v18.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v19.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v20.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v20.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v21.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v21.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v20.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v21.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v22.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v22.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v23.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v23.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v22.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v23.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v24.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v24.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v25.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v25.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v24.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v25.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v26.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v26.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v27.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v27.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v26.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v27.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v28.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v28.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v29.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v29.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v28.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v29.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "fcmge  v4.4s,    v30.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v5.4s,    v30.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "fcmge  v6.4s,    v31.4s,    v2.4s   \n" /* vcgeq_f32 */  \
  "fmul   v7.4s,    v31.4s,    v3.4s   \n" /* vmulq_f32 */  \
  "bif    v30.16b,   v5.16b,   v4.16b  \n" /* choose*/      \
  "bif    v31.16b,   v7.16b,   v6.16b  \n" /* choose*/      \
  "b 12f \n"

#define GEMM_SDOT_HARD_SWISH_8x4                            \
  "15: \n"                                                  \
  "ldr    q4,      [%[alpha], #16]    \n" /* offset */      \
  "ld1    {v3.4s}, [%[alpha]]\n"   /* leakyrelu alpha */    \
  "ldr    q5,      [%[alpha], #32]    \n" /* theshold */    \
  "fadd   v6.4s,    v8.4s,    v4.4s   \n"                   \
  "fadd   v7.4s,    v11.4s,   v4.4s   \n"                   \
  "fmul   v8.4s,    v8.4s,    v3.4s   \n"                   \
  "fmul   v11.4s,   v11.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,   v5.4s    \n"                   \
  "fmul   v8.4s,    v8.4s,    v6.4s   \n"                   \
  "fmul   v11.4s,   v11.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v14.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v17.4s,   v4.4s   \n"                   \
  "fmul   v14.4s,   v14.4s,   v3.4s  \n"                    \
  "fmul   v17.4s,   v17.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v14.4s,   v14.4s,   v6.4s   \n"                   \
  "fmul   v17.4s,   v17.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v20.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v23.4s,   v4.4s   \n"                   \
  "fmul   v20.4s,   v20.4s,   v3.4s  \n"                    \
  "fmul   v23.4s,   v23.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v20.4s,   v20.4s,   v6.4s   \n"                   \
  "fmul   v23.4s,   v23.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v26.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v29.4s,   v4.4s   \n"                   \
  "fmul   v26.4s,   v26.4s,   v3.4s  \n"                    \
  "fmul   v29.4s,   v29.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v26.4s,   v26.4s,   v6.4s   \n"                   \
  "fmul   v29.4s,   v29.4s,   v7.4s   \n"                   \
  "12:  \n"

#define GEMM_SDOT_HARD_SWISH_8x8                            \
  "15: \n"                                                  \
  "ldr    q4,      [%[alpha], #16]    \n" /* offset */      \
  "ld1    {v3.4s}, [%[alpha]]\n"   /* leakyrelu alpha */    \
  "ldr    q5,      [%[alpha], #32]    \n" /* theshold */    \
  "fadd   v6.4s,    v8.4s,    v4.4s   \n"                   \
  "fadd   v7.4s,    v11.4s,   v4.4s   \n"                   \
  "fmul   v8.4s,    v8.4s,    v3.4s   \n"                   \
  "fmul   v11.4s,   v11.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,   v5.4s    \n"                   \
  "fmul   v8.4s,    v8.4s,    v6.4s   \n"                   \
  "fmul   v11.4s,   v11.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v14.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v17.4s,   v4.4s   \n"                   \
  "fmul   v14.4s,   v14.4s,   v3.4s  \n"                    \
  "fmul   v17.4s,   v17.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v14.4s,   v14.4s,   v6.4s   \n"                   \
  "fmul   v17.4s,   v17.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v20.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v23.4s,   v4.4s   \n"                   \
  "fmul   v20.4s,   v20.4s,   v3.4s  \n"                    \
  "fmul   v23.4s,   v23.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v20.4s,   v20.4s,   v6.4s   \n"                   \
  "fmul   v23.4s,   v23.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v26.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v29.4s,   v4.4s   \n"                   \
  "fmul   v26.4s,   v26.4s,   v3.4s  \n"                    \
  "fmul   v29.4s,   v29.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v26.4s,   v26.4s,   v6.4s   \n"                   \
  "fmul   v29.4s,   v29.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v9.4s,    v4.4s   \n"                   \
  "fadd   v7.4s,    v12.4s,   v4.4s   \n"                   \
  "fmul   v9.4s,    v9.4s,    v3.4s   \n"                   \
  "fmul   v12.4s,   v12.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,   v5.4s    \n"                   \
  "fmul   v9.4s,    v9.4s,    v6.4s   \n"                   \
  "fmul   v12.4s,   v12.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v15.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v18.4s,   v4.4s   \n"                   \
  "fmul   v15.4s,   v15.4s,   v3.4s  \n"                    \
  "fmul   v18.4s,   v18.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v15.4s,   v15.4s,   v6.4s   \n"                   \
  "fmul   v18.4s,   v18.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v21.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v24.4s,   v4.4s   \n"                   \
  "fmul   v21.4s,   v21.4s,   v3.4s  \n"                    \
  "fmul   v24.4s,   v24.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v21.4s,   v21.4s,   v6.4s   \n"                   \
  "fmul   v24.4s,   v24.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v27.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v30.4s,   v4.4s   \n"                   \
  "fmul   v27.4s,   v27.4s,   v3.4s  \n"                    \
  "fmul   v30.4s,   v30.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v27.4s,   v27.4s,   v6.4s   \n"                   \
  "fmul   v30.4s,   v30.4s,   v7.4s   \n"                   \
  "12:  \n"

#define GEMM_SDOT_HARD_SWISH                        \
  "15: \n"                                                  \
  "ldr    q4,      [%[alpha], #16]    \n" /* offset */      \
  "ld1    {v3.4s}, [%[alpha]]\n"   /* leakyrelu alpha */    \
  "ldr    q5,      [%[alpha], #32]    \n" /* theshold */    \
  "fadd   v6.4s,    v8.4s,    v4.4s   \n"                   \
  "fadd   v7.4s,    v11.4s,   v4.4s   \n"                   \
  "fmul   v8.4s,    v8.4s,    v3.4s   \n"                   \
  "fmul   v11.4s,   v11.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,   v5.4s    \n"                   \
  "fmul   v8.4s,    v8.4s,    v6.4s   \n"                   \
  "fmul   v11.4s,   v11.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v14.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v17.4s,   v4.4s   \n"                   \
  "fmul   v14.4s,   v14.4s,   v3.4s  \n"                    \
  "fmul   v17.4s,   v17.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v14.4s,   v14.4s,   v6.4s   \n"                   \
  "fmul   v17.4s,   v17.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v20.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v23.4s,   v4.4s   \n"                   \
  "fmul   v20.4s,   v20.4s,   v3.4s  \n"                    \
  "fmul   v23.4s,   v23.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v20.4s,   v20.4s,   v6.4s   \n"                   \
  "fmul   v23.4s,   v23.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v26.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v29.4s,   v4.4s   \n"                   \
  "fmul   v26.4s,   v26.4s,   v3.4s  \n"                    \
  "fmul   v29.4s,   v29.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v26.4s,   v26.4s,   v6.4s   \n"                   \
  "fmul   v29.4s,   v29.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v9.4s,    v4.4s   \n"                   \
  "fadd   v7.4s,    v12.4s,   v4.4s   \n"                   \
  "fmul   v9.4s,    v9.4s,    v3.4s   \n"                   \
  "fmul   v12.4s,   v12.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,   v5.4s    \n"                   \
  "fmul   v9.4s,    v9.4s,    v6.4s   \n"                   \
  "fmul   v12.4s,   v12.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v15.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v18.4s,   v4.4s   \n"                   \
  "fmul   v15.4s,   v15.4s,   v3.4s  \n"                    \
  "fmul   v18.4s,   v18.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v15.4s,   v15.4s,   v6.4s   \n"                   \
  "fmul   v18.4s,   v18.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v21.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v24.4s,   v4.4s   \n"                   \
  "fmul   v21.4s,   v21.4s,   v3.4s  \n"                    \
  "fmul   v24.4s,   v24.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v21.4s,   v21.4s,   v6.4s   \n"                   \
  "fmul   v24.4s,   v24.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v27.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v30.4s,   v4.4s   \n"                   \
  "fmul   v27.4s,   v27.4s,   v3.4s  \n"                    \
  "fmul   v30.4s,   v30.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v27.4s,   v27.4s,   v6.4s   \n"                   \
  "fmul   v30.4s,   v30.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v10.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v13.4s,   v4.4s   \n"                   \
  "fmul   v10.4s,   v10.4s,   v3.4s   \n"                   \
  "fmul   v13.4s,   v13.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s    \n"                  \
  "fmul   v10.4s,   v10.4s,   v6.4s   \n"                   \
  "fmul   v13.4s,   v13.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v16.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v19.4s,   v4.4s   \n"                   \
  "fmul   v16.4s,   v16.4s,   v3.4s  \n"                    \
  "fmul   v19.4s,   v19.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v16.4s,   v16.4s,   v6.4s   \n"                   \
  "fmul   v19.4s,   v19.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v22.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v25.4s,   v4.4s   \n"                   \
  "fmul   v22.4s,   v22.4s,   v3.4s  \n"                    \
  "fmul   v25.4s,   v25.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v22.4s,   v22.4s,   v6.4s   \n"                   \
  "fmul   v25.4s,   v25.4s,   v7.4s   \n"                   \
  "fadd   v6.4s,    v28.4s,   v4.4s   \n"                   \
  "fadd   v7.4s,    v31.4s,   v4.4s   \n"                   \
  "fmul   v28.4s,   v28.4s,   v3.4s  \n"                    \
  "fmul   v31.4s,   v31.4s,   v3.4s   \n"                   \
  "fmax   v6.4s,    v6.4s,    v2.4s   \n"                   \
  "fmax   v7.4s,    v7.4s,    v2.4s   \n"                   \
  "fmin   v6.4s,    v6.4s,    v5.4s   \n"                   \
  "fmin   v7.4s,    v7.4s,    v5.4s   \n"                   \
  "fmul   v28.4s,   v28.4s,   v6.4s   \n"                   \
  "fmul   v31.4s,   v31.4s,   v7.4s   \n"                   \
  "12:  \n"

#define GEMM_SDOT_CVT_INT32_TO_FP32_8x4                                    \
  "ldp  q0, q1, [%[scale]]\n"     /* load scale */                         \
  "ldp  q2, q3, [%[bias_ptr]]\n"  /* load bias */                          \
  "scvtf  v4.4s , v8.4s\n"        /*  00, convert to fp32 */               \
  "dup    v8.4s,  v2.s[0]\n"      /*  fill with bias*/                     \
  "fmla v8.4s, v4.4s, v0.s[0]\n"  /*  00, mul scale to get final result */ \
  "scvtf  v4.4s , v11.4s\n"       /*  10, convert to fp32 */               \
  "dup    v11.4s, v2.s[1]\n"      /*  fill with bias*/                     \
  "fmla v11.4s, v4.4s, v0.s[1]\n" /*  10, mul scale to get final result */ \
  "scvtf  v4.4s , v14.4s\n"       /*  20, convert to fp32 */               \
  "dup    v14.4s, v2.s[2]\n"      /*  fill with bias*/                     \
  "fmla v14.4s, v4.4s, v0.s[2]\n" /*  20, mul scale to get final result */ \
  "scvtf  v4.4s , v17.4s\n"       /*  30, convert to fp32 */               \
  "dup    v17.4s, v2.s[3]\n"      /*  fill with bias*/                     \
  "fmla v17.4s, v4.4s, v0.s[3]\n" /*  30, mul scale to get final result */ \
  "scvtf  v4.4s , v20.4s\n"       /*  40, convert to fp32 */               \
  "dup    v20.4s, v3.s[0]\n"      /*  fill with bias*/                     \
  "fmla v20.4s, v4.4s, v1.s[0]\n" /*  40, mul scale to get final result */ \
  "scvtf  v4.4s , v23.4s\n"       /*  50, convert to fp32 */               \
  "dup    v23.4s, v3.s[1]\n"      /*  fill with bias*/                     \
  "fmla v23.4s, v4.4s, v1.s[1]\n" /*  50, mul scale to get final result */ \
  "scvtf  v4.4s , v26.4s\n"       /*  60, convert to fp32 */               \
  "dup    v26.4s, v3.s[2]\n"      /*  fill with bias*/                     \
  "fmla v26.4s, v4.4s, v1.s[2]\n" /*  60, mul scale to get final result */ \
  "scvtf  v4.4s, v29.4s\n"        /*  70, convert to fp32 */               \
  "dup    v29.4s, v3.s[3]\n"      /*  fill with bias*/                     \
  "fmla v29.4s, v4.4s,v1.s[3]\n"  /*  70, mul scale to get final result */

#define GEMM_SDOT_CVT_INT32_TO_FP32_8x8                                    \
  "ldp  q0, q1, [%[scale]]\n"     /* load scale */                         \
  "ldp  q2, q3, [%[bias_ptr]]\n"  /* load bias */                          \
  "scvtf  v4.4s , v8.4s\n"        /*  00, convert to fp32 */               \
  "scvtf  v5.4s , v9.4s\n"        /*  01, convert to fp32 */               \
  "dup    v8.4s,  v2.s[0]\n"      /*  fill with bias*/                     \
  "dup    v9.4s,  v2.s[0]\n"      /*  fill with bias*/                     \
  "fmla v8.4s, v4.4s, v0.s[0]\n"  /*  00, mul scale to get final result */ \
  "fmla v9.4s, v5.4s, v0.s[0]\n"  /*  01, mul scale to get final result */ \
  "scvtf  v4.4s , v11.4s\n"       /*  10, convert to fp32 */               \
  "scvtf  v5.4s , v12.4s\n"       /*  11, convert to fp32 */               \
  "dup    v11.4s, v2.s[1]\n"      /*  fill with bias*/                     \
  "dup    v12.4s, v2.s[1]\n"      /*  fill with bias*/                     \
  "fmla v11.4s, v4.4s, v0.s[1]\n" /*  10, mul scale to get final result */ \
  "fmla v12.4s, v5.4s, v0.s[1]\n" /*  11, mul scale to get final result */ \
  "scvtf  v4.4s , v14.4s\n"       /*  20, convert to fp32 */               \
  "scvtf  v5.4s , v15.4s\n"       /*  21, convert to fp32 */               \
  "dup    v14.4s, v2.s[2]\n"      /*  fill with bias*/                     \
  "dup    v15.4s, v2.s[2]\n"      /*  fill with bias*/                     \
  "fmla v14.4s, v4.4s, v0.s[2]\n" /*  20, mul scale to get final result */ \
  "fmla v15.4s, v5.4s, v0.s[2]\n" /*  21, mul scale to get final result */ \
  "scvtf  v4.4s , v17.4s\n"       /*  30, convert to fp32 */               \
  "scvtf  v5.4s , v18.4s\n"       /*  31, convert to fp32 */               \
  "dup    v17.4s, v2.s[3]\n"      /*  fill with bias*/                     \
  "dup    v18.4s, v2.s[3]\n"      /*  fill with bias*/                     \
  "fmla v17.4s, v4.4s, v0.s[3]\n" /*  30, mul scale to get final result */ \
  "fmla v18.4s, v5.4s, v0.s[3]\n" /*  31, mul scale to get final result */ \
  "scvtf  v4.4s , v20.4s\n"       /*  40, convert to fp32 */               \
  "scvtf  v5.4s , v21.4s\n"       /*  41, convert to fp32 */               \
  "dup    v20.4s, v3.s[0]\n"      /*  fill with bias*/                     \
  "dup    v21.4s, v3.s[0]\n"      /*  fill with bias*/                     \
  "fmla v20.4s, v4.4s, v1.s[0]\n" /*  40, mul scale to get final result */ \
  "fmla v21.4s, v5.4s, v1.s[0]\n" /*  41, mul scale to get final result */ \
  "scvtf  v4.4s , v23.4s\n"       /*  50, convert to fp32 */               \
  "scvtf  v5.4s , v24.4s\n"       /*  51, convert to fp32 */               \
  "dup    v23.4s, v3.s[1]\n"      /*  fill with bias*/                     \
  "dup    v24.4s, v3.s[1]\n"      /*  fill with bias*/                     \
  "fmla v23.4s, v4.4s, v1.s[1]\n" /*  50, mul scale to get final result */ \
  "fmla v24.4s, v5.4s, v1.s[1]\n" /*  51, mul scale to get final result */ \
  "scvtf  v4.4s , v26.4s\n"       /*  60, convert to fp32 */               \
  "scvtf  v5.4s , v27.4s\n"       /*  61, convert to fp32 */               \
  "dup    v26.4s, v3.s[2]\n"      /*  fill with bias*/                     \
  "dup    v27.4s, v3.s[2]\n"      /*  fill with bias*/                     \
  "fmla v26.4s, v4.4s, v1.s[2]\n" /*  60, mul scale to get final result */ \
  "fmla v27.4s, v5.4s, v1.s[2]\n" /*  61, mul scale to get final result */ \
  "scvtf  v4.4s, v29.4s\n"        /*  70, convert to fp32 */               \
  "scvtf  v5.4s, v30.4s\n"        /*  71, convert to fp32 */               \
  "dup    v29.4s, v3.s[3]\n"      /*  fill with bias*/                     \
  "dup    v30.4s, v3.s[3]\n"      /*  fill with bias*/                     \
  "fmla v29.4s, v4.4s,v1.s[3]\n"  /*  70, mul scale to get final result */ \
  "fmla v30.4s, v5.4s,v1.s[3]\n"  /*  71, mul scale to get final result */

#define GEMM_SDOT_CVT_INT32_TO_FP32                                        \
  "ldp  q0, q1, [%[scale]]\n"     /* load scale */                         \
  "ldp  q2, q3, [%[bias_ptr]]\n"  /* load bias */                          \
  "scvtf  v4.4s , v8.4s\n"        /*  00, convert to fp32 */               \
  "scvtf  v5.4s , v9.4s\n"        /*  01, convert to fp32 */               \
  "scvtf  v6.4s , v10.4s\n"       /*  02, convert to fp32 */               \
  "dup    v8.4s,  v2.s[0]\n"      /*  fill with bias*/                     \
  "dup    v9.4s,  v2.s[0]\n"      /*  fill with bias*/                     \
  "dup    v10.4s,  v2.s[0]\n"     /*  fill with bias*/                     \
  "fmla v8.4s, v4.4s, v0.s[0]\n"  /*  00, mul scale to get final result */ \
  "fmla v9.4s, v5.4s, v0.s[0]\n"  /*  01, mul scale to get final result */ \
  "fmla v10.4s, v6.4s, v0.s[0]\n" /*  02, mul scale to get final result */ \
  "scvtf  v4.4s , v11.4s\n"       /*  10, convert to fp32 */               \
  "scvtf  v5.4s , v12.4s\n"       /*  11, convert to fp32 */               \
  "scvtf  v6.4s , v13.4s\n"       /*  12, convert to fp32 */               \
  "dup    v11.4s, v2.s[1]\n"      /*  fill with bias*/                     \
  "dup    v12.4s, v2.s[1]\n"      /*  fill with bias*/                     \
  "dup    v13.4s, v2.s[1]\n"      /*  fill with bias*/                     \
  "fmla v11.4s, v4.4s, v0.s[1]\n" /*  10, mul scale to get final result */ \
  "fmla v12.4s, v5.4s, v0.s[1]\n" /*  11, mul scale to get final result */ \
  "fmla v13.4s, v6.4s, v0.s[1]\n" /*  12, mul scale to get final result */ \
  "scvtf  v4.4s , v14.4s\n"       /*  20, convert to fp32 */               \
  "scvtf  v5.4s , v15.4s\n"       /*  21, convert to fp32 */               \
  "scvtf  v6.4s , v16.4s\n"       /*  22, convert to fp32 */               \
  "dup    v14.4s, v2.s[2]\n"      /*  fill with bias*/                     \
  "dup    v15.4s, v2.s[2]\n"      /*  fill with bias*/                     \
  "dup    v16.4s, v2.s[2]\n"      /*  fill with bias*/                     \
  "fmla v14.4s, v4.4s, v0.s[2]\n" /*  20, mul scale to get final result */ \
  "fmla v15.4s, v5.4s, v0.s[2]\n" /*  21, mul scale to get final result */ \
  "fmla v16.4s, v6.4s, v0.s[2]\n" /*  22, mul scale to get final result */ \
  "scvtf  v4.4s , v17.4s\n"       /*  30, convert to fp32 */               \
  "scvtf  v5.4s , v18.4s\n"       /*  31, convert to fp32 */               \
  "scvtf  v6.4s , v19.4s\n"       /*  32, convert to fp32 */               \
  "dup    v17.4s, v2.s[3]\n"      /*  fill with bias*/                     \
  "dup    v18.4s, v2.s[3]\n"      /*  fill with bias*/                     \
  "dup    v19.4s, v2.s[3]\n"      /*  fill with bias*/                     \
  "fmla v17.4s, v4.4s, v0.s[3]\n" /*  30, mul scale to get final result */ \
  "fmla v18.4s, v5.4s, v0.s[3]\n" /*  31, mul scale to get final result */ \
  "fmla v19.4s, v6.4s, v0.s[3]\n" /*  32, mul scale to get final result */ \
  "scvtf  v4.4s , v20.4s\n"       /*  40, convert to fp32 */               \
  "scvtf  v5.4s , v21.4s\n"       /*  41, convert to fp32 */               \
  "scvtf  v6.4s , v22.4s\n"       /*  42, convert to fp32 */               \
  "dup    v20.4s, v3.s[0]\n"      /*  fill with bias*/                     \
  "dup    v21.4s, v3.s[0]\n"      /*  fill with bias*/                     \
  "dup    v22.4s, v3.s[0]\n"      /*  fill with bias*/                     \
  "fmla v20.4s, v4.4s, v1.s[0]\n" /*  40, mul scale to get final result */ \
  "fmla v21.4s, v5.4s, v1.s[0]\n" /*  41, mul scale to get final result */ \
  "fmla v22.4s, v6.4s, v1.s[0]\n" /*  42, mul scale to get final result */ \
  "scvtf  v4.4s , v23.4s\n"       /*  50, convert to fp32 */               \
  "scvtf  v5.4s , v24.4s\n"       /*  51, convert to fp32 */               \
  "scvtf  v6.4s , v25.4s\n"       /*  52, convert to fp32 */               \
  "dup    v23.4s, v3.s[1]\n"      /*  fill with bias*/                     \
  "dup    v24.4s, v3.s[1]\n"      /*  fill with bias*/                     \
  "dup    v25.4s, v3.s[1]\n"      /*  fill with bias*/                     \
  "fmla v23.4s, v4.4s, v1.s[1]\n" /*  50, mul scale to get final result */ \
  "fmla v24.4s, v5.4s, v1.s[1]\n" /*  51, mul scale to get final result */ \
  "fmla v25.4s, v6.4s, v1.s[1]\n" /*  52, mul scale to get final result */ \
  "scvtf  v4.4s , v26.4s\n"       /*  60, convert to fp32 */               \
  "scvtf  v5.4s , v27.4s\n"       /*  61, convert to fp32 */               \
  "scvtf  v6.4s , v28.4s\n"       /*  62, convert to fp32 */               \
  "dup    v26.4s, v3.s[2]\n"      /*  fill with bias*/                     \
  "dup    v27.4s, v3.s[2]\n"      /*  fill with bias*/                     \
  "dup    v28.4s, v3.s[2]\n"      /*  fill with bias*/                     \
  "fmla v26.4s, v4.4s, v1.s[2]\n" /*  60, mul scale to get final result */ \
  "fmla v27.4s, v5.4s, v1.s[2]\n" /*  61, mul scale to get final result */ \
  "fmla v28.4s, v6.4s, v1.s[2]\n" /*  62, mul scale to get final result */ \
  "scvtf  v4.4s, v29.4s\n"        /*  70, convert to fp32 */               \
  "scvtf  v5.4s, v30.4s\n"        /*  71, convert to fp32 */               \
  "scvtf  v6.4s, v31.4s\n"        /*  72, convert to fp32 */               \
  "dup    v29.4s, v3.s[3]\n"      /*  fill with bias*/                     \
  "dup    v30.4s, v3.s[3]\n"      /*  fill with bias*/                     \
  "dup    v31.4s, v3.s[3]\n"      /*  fill with bias*/                     \
  "fmla v29.4s, v4.4s,v1.s[3]\n"  /*  70, mul scale to get final result */ \
  "fmla v30.4s, v5.4s,v1.s[3]\n"  /*  71, mul scale to get final result */ \
  "fmla v31.4s, v6.4s,v1.s[3]\n"  /*  72, mul scale to get final result */

#define GEMM_SDOT_FP32_OUT_8x4                                \
  GEMM_SDOT_CVT_INT32_TO_FP32_8x4                             \
  GEMM_SDOT_RELU_8x4                                          \
  GEMM_SDOT_RELU6_8x4                                         \
  GEMM_SDOT_LEAKY_RELU_8x4                                    \
  GEMM_SDOT_HARD_SWISH_8x4                                    \
  "st1 {v8.4s},[%[c_ptr0]],  #16\n" /* store r0 */            \
  "st1 {v11.4s},[%[c_ptr1]], #16\n" /* store r1 */            \
  "st1 {v14.4s},[%[c_ptr2]], #16\n" /* store r2 */            \
  "st1 {v17.4s},[%[c_ptr3]], #16\n" /* store r3 */            \
  "st1 {v20.4s},[%[c_ptr4]], #16\n" /* store r4 */            \
  "st1 {v23.4s},[%[c_ptr5]], #16\n" /* store r5 */            \
  "st1 {v26.4s},[%[c_ptr6]], #16\n" /* store r6 */            \
  "st1 {v29.4s},[%[c_ptr7]], #16\n" /* store r7 */

#define GEMM_SDOT_FP32_OUT_8x8                                \
  GEMM_SDOT_CVT_INT32_TO_FP32_8x8                             \
  GEMM_SDOT_RELU_8x8                                          \
  GEMM_SDOT_RELU6_8x8                                         \
  GEMM_SDOT_LEAKY_RELU_8x8                                    \
  GEMM_SDOT_HARD_SWISH_8x8                                    \
  "st1 {v8.4s, v9.4s},[%[c_ptr0]],   #32\n"   /* store r0 */  \
  "st1 {v11.4s, v12.4s},[%[c_ptr1]], #32\n" /* store r1 */    \
  "st1 {v14.4s, v15.4s},[%[c_ptr2]], #32\n" /* store r2 */    \
  "st1 {v17.4s, v18.4s},[%[c_ptr3]], #32\n" /* store r3 */    \
  "st1 {v20.4s, v21.4s},[%[c_ptr4]], #32\n" /* store r4 */    \
  "st1 {v23.4s, v24.4s},[%[c_ptr5]], #32\n" /* store r5 */    \
  "st1 {v26.4s, v27.4s},[%[c_ptr6]], #32\n" /* store r6 */    \
  "st1 {v29.4s, v30.4s},[%[c_ptr7]], #32\n" /* store r7 */

#define GEMM_SDOT_FP32_OUT                                         \
  GEMM_SDOT_CVT_INT32_TO_FP32                                      \
  GEMM_SDOT_RELU                                                   \
  GEMM_SDOT_RELU6                                                  \
  GEMM_SDOT_LEAKY_RELU                                             \
  GEMM_SDOT_HARD_SWISH                                             \
  "st1 {v8.4s, v9.4s, v10.4s},[%[c_ptr0]], #48\n"   /* store r0 */ \
  "st1 {v11.4s, v12.4s, v13.4s},[%[c_ptr1]], #48\n" /* store r1 */ \
  "st1 {v14.4s, v15.4s, v16.4s},[%[c_ptr2]], #48\n" /* store r2 */ \
  "st1 {v17.4s, v18.4s, v19.4s},[%[c_ptr3]], #48\n" /* store r3 */ \
  "st1 {v20.4s, v21.4s, v22.4s},[%[c_ptr4]], #48\n" /* store r4 */ \
  "st1 {v23.4s, v24.4s, v25.4s},[%[c_ptr5]], #48\n" /* store r5 */ \
  "st1 {v26.4s, v27.4s, v28.4s},[%[c_ptr6]], #48\n" /* store r6 */ \
  "st1 {v29.4s, v30.4s, v31.4s},[%[c_ptr7]], #48\n" /* store r7 */

#define GEMM_SDOT_INT8_OUT_8x4                                     \
  GEMM_SDOT_CVT_INT32_TO_FP32_8x4                                  \
  GEMM_SDOT_RELU_8x4                                               \
  GEMM_SDOT_RELU6_8x4                                              \
  GEMM_SDOT_LEAKY_RELU_8x4                                         \
  GEMM_SDOT_HARD_SWISH_8x4                                         \
  "ld1  {v6.4s}, [%[vmax]]\n"     /* v8 = -127.f     */            \
  /* data >= -127 */                                               \
  "fcmge v0.4s, v8.4s, v6.4s\n"                                    \
  "fcmge v3.4s, v11.4s, v6.4s\n"                                   \
  "fcmge v7.4s, v14.4s, v6.4s\n"                                   \
  /* choose data */                                                \
  "bif v8.16b, v6.16b, v0.16b\n"                                   \
  "bif v11.16b, v6.16b, v3.16b\n"                                  \
  "bif v14.16b, v6.16b, v7.16b\n"                                  \
  "fcvtas v0.4s, v8.4s\n"         /*  00, cvt to int */            \
  "fcvtas v3.4s, v11.4s\n"        /*  10, cvt to int */            \
  "fcvtas v6.4s, v14.4s\n"        /*  20, cvt to int */            \
  "sqxtn  v10.4h, v0.4s\n"        /*  00, cvt int32 to int16 */    \
  "sqxtn  v12.4h, v3.4s\n"        /*  10, cvt int32 to int16 */    \
  "sqxtn  v14.4h, v6.4s\n"        /*  20, cvt int32 to int16 */    \
  "ld1  {v6.4s}, [%[vmax]]\n"     /* v8 = -127.f     */            \
  /* data >= -127 */                                               \
  "fcmge v1.4s, v17.4s, v6.4s\n"                                   \
  "fcmge v4.4s, v20.4s, v6.4s\n"                                   \
  "fcmge v8.4s, v23.4s, v6.4s\n"                                   \
  /* choose data */                                                \
  "bif v17.16b, v6.16b, v1.16b\n"                                  \
  "bif v20.16b, v6.16b, v4.16b\n"                                  \
  "bif v23.16b, v6.16b, v8.16b\n"                                  \
  "fcvtas v1.4s, v17.4s\n"        /*  30, cvt to int */            \
  "fcvtas v4.4s, v20.4s\n"        /*  40, cvt to int */            \
  "fcvtas v7.4s, v23.4s\n"        /*  50, cvt to int */            \
  "sqxtn  v16.4h, v1.4s\n"        /*  30, cvt int32 to int16 */    \
  "sqxtn  v18.4h, v4.4s\n"        /*  40, cvt int32 to int16 */    \
  "sqxtn  v20.4h, v7.4s\n"        /*  50, cvt int32 to int16 */    \
  "ld1  {v6.4s}, [%[vmax]]\n"     /* v8 = -127.f     */            \
  /* data >= -127 */                                               \
  "fcmge v0.4s, v26.4s, v6.4s\n"                                   \
  "fcmge v3.4s, v29.4s, v6.4s\n"                                   \
  /* choose data */                                                \
  "bif v26.16b, v6.16b, v0.16b\n"                                  \
  "bif v29.16b, v6.16b, v3.16b\n"                                  \
  "fcvtas v0.4s, v26.4s\n"        /*  60, cvt to int */         \
  "fcvtas v3.4s, v29.4s\n"        /*  70, cvt to int */         \
  "sqxtn  v22.4h, v0.4s\n"        /*  60, cvt int32 to int16 */ \
  "sqxtn  v24.4h, v3.4s\n"        /*  70, cvt int32 to int16 */ \
  "sqxtn  v0.8b, v10.8h\n"        /*  00, cvt int16 to int8 */  \
  "sqxtn  v1.8b, v12.8h\n"        /*  10, cvt int16 to int8 */  \
  "sqxtn  v2.8b, v14.8h\n"        /*  20, cvt int16 to int8 */  \
  "sqxtn  v3.8b, v16.8h\n"        /*  30, cvt int16 to int8 */  \
  "sqxtn  v4.8b, v18.8h\n"        /*  40, cvt int16 to int8 */  \
  "sqxtn  v5.8b, v20.8h\n"        /*  50, cvt int16 to int8 */  \
  "sqxtn  v6.8b, v22.8h\n"        /*  60, cvt int16 to int8 */  \
  "sqxtn  v7.8b, v24.8h\n"        /*  70, cvt int16 to int8 */  \
  "str s0,[%[c_ptr0]], #4\n"      /* store r0 */                \
  "str s1,[%[c_ptr1]], #4\n"      /* store r1 */                \
  "str s2,[%[c_ptr2]], #4\n"     /* store r2 */                 \
  "str s3,[%[c_ptr3]], #4\n"     /* store r3 */                 \
  "str s4,[%[c_ptr4]], #4\n"     /* store r4 */                 \
  "str s5,[%[c_ptr5]], #4\n"     /* store r5 */                 \
  "str s6,[%[c_ptr6]], #4\n"     /* store r6 */                 \
  "str s7,[%[c_ptr7]], #4\n"     /* store r7 */

#define GEMM_SDOT_INT8_OUT_8x8                                  \
  GEMM_SDOT_CVT_INT32_TO_FP32_8x8                               \
  GEMM_SDOT_RELU_8x8                                            \
  GEMM_SDOT_RELU6_8x8                                           \
  GEMM_SDOT_LEAKY_RELU_8x8                                      \
  GEMM_SDOT_HARD_SWISH_8x8                                      \
  "ld1  {v6.4s}, [%[vmax]]\n"     /* v8 = -127.f     */         \
  /* data >= -127 */                                            \
  "fcmge v0.4s, v8.4s, v6.4s\n"                                 \
  "fcmge v1.4s, v9.4s, v6.4s\n"                                 \
  "fcmge v3.4s, v11.4s, v6.4s\n"                                \
  "fcmge v4.4s, v12.4s, v6.4s\n"                                \
  "fcmge v7.4s, v14.4s, v6.4s\n"                                \
  /* choose data */                                             \
  "bif v8.16b, v6.16b, v0.16b\n"                                \
  "fcmge v0.4s, v15.4s, v6.4s\n"                                \
  "bif v9.16b, v6.16b, v1.16b\n"                                \
  "bif v11.16b, v6.16b, v3.16b\n"                               \
  "bif v12.16b, v6.16b, v4.16b\n"                               \
  "bif v14.16b, v6.16b, v7.16b\n"                               \
  "bif v15.16b, v6.16b, v0.16b \n"                              \
  "fcvtas v0.4s, v8.4s\n"         /*  00, cvt to int */         \
  "fcvtas v1.4s, v9.4s\n"         /*  01, cvt to int */         \
  "fcvtas v3.4s, v11.4s\n"        /*  10, cvt to int */         \
  "fcvtas v4.4s, v12.4s\n"        /*  11, cvt to int */         \
  "fcvtas v6.4s, v14.4s\n"        /*  20, cvt to int */         \
  "fcvtas v7.4s, v15.4s\n"        /*  21, cvt to int */         \
  "sqxtn  v10.4h, v0.4s\n"        /*  00, cvt int32 to int16 */ \
  "sqxtn2 v10.8h, v1.4s\n"        /*  01, cvt int32 to int16 */ \
  "sqxtn  v12.4h, v3.4s\n"        /*  10, cvt int32 to int16 */ \
  "sqxtn2 v12.8h, v4.4s\n"        /*  11, cvt int32 to int16 */ \
  "sqxtn  v14.4h, v6.4s\n"        /*  20, cvt int32 to int16 */ \
  "ld1  {v6.4s}, [%[vmax]]\n"     /* v8 = -127.f     */         \
  "sqxtn2 v14.8h, v7.4s\n"        /*  21, cvt int32 to int16 */ \
  /* data >= -127 */                                               \
  "fcmge v1.4s, v17.4s, v6.4s\n"                                   \
  "fcmge v2.4s, v18.4s, v6.4s\n"                                   \
  "fcmge v4.4s, v20.4s, v6.4s\n"                                   \
  "fcmge v5.4s, v21.4s, v6.4s\n"                                   \
  "fcmge v8.4s, v23.4s, v6.4s\n"                                   \
  "fcmge v9.4s, v24.4s, v6.4s\n"                                   \
  /* choose data */                                                \
  "bif v17.16b, v6.16b, v1.16b\n"                                  \
  "bif v18.16b, v6.16b, v2.16b\n"                                  \
  "bif v20.16b, v6.16b, v4.16b\n"                                  \
  "bif v21.16b, v6.16b, v5.16b\n"                                  \
  "bif v23.16b, v6.16b, v8.16b\n"                                  \
  "bif v24.16b, v6.16b, v9.16b\n"                                  \
  "fcvtas v1.4s, v17.4s\n"        /*  30, cvt to int */         \
  "fcvtas v2.4s, v18.4s\n"        /*  31, cvt to int */         \
  "fcvtas v4.4s, v20.4s\n"        /*  40, cvt to int */         \
  "fcvtas v5.4s, v21.4s\n"        /*  41, cvt to int */         \
  "fcvtas v7.4s, v23.4s\n"        /*  50, cvt to int */         \
  "fcvtas v8.4s, v24.4s\n"        /*  51, cvt to int */         \
  "sqxtn  v16.4h, v1.4s\n"        /*  30, cvt int32 to int16 */ \
  "sqxtn2 v16.8h, v2.4s\n"        /*  31, cvt int32 to int16 */ \
  "sqxtn  v18.4h, v4.4s\n"        /*  40, cvt int32 to int16 */ \
  "sqxtn2 v18.8h, v5.4s\n"        /*  41, cvt int32 to int16 */ \
  "sqxtn  v20.4h, v7.4s\n"        /*  50, cvt int32 to int16 */ \
  "sqxtn2 v20.8h, v8.4s\n"        /*  51, cvt int32 to int16 */ \
  "ld1  {v6.4s}, [%[vmax]]\n"     /* v8 = -127.f     */            \
  /* data >= -127 */                                               \
  "fcmge v0.4s, v26.4s, v6.4s\n"                                   \
  "fcmge v1.4s, v27.4s, v6.4s\n"                                   \
  "fcmge v3.4s, v29.4s, v6.4s\n"                                   \
  "fcmge v4.4s, v30.4s, v6.4s\n"                                   \
  /* choose data */                                                \
  "bif v26.16b, v6.16b, v0.16b\n"                                  \
  "bif v27.16b, v6.16b, v1.16b\n"                                  \
  "bif v29.16b, v6.16b, v3.16b\n"                                  \
  "bif v30.16b, v6.16b, v4.16b\n"                                  \
  "fcvtas v0.4s, v26.4s\n"        /*  60, cvt to int */         \
  "fcvtas v1.4s, v27.4s\n"        /*  61, cvt to int */         \
  "fcvtas v3.4s, v29.4s\n"        /*  70, cvt to int */         \
  "fcvtas v4.4s, v30.4s\n"        /*  71, cvt to int */         \
  "sqxtn  v22.4h, v0.4s\n"        /*  60, cvt int32 to int16 */ \
  "sqxtn2 v22.8h, v1.4s\n"        /*  61, cvt int32 to int16 */ \
  "sqxtn  v24.4h, v3.4s\n"        /*  70, cvt int32 to int16 */ \
  "sqxtn2 v24.8h, v4.4s\n"        /*  71, cvt int32 to int16 */ \
  "sqxtn  v0.8b, v10.8h\n"        /*  00, cvt int16 to int8 */  \
  "sqxtn  v1.8b, v12.8h\n"        /*  10, cvt int16 to int8 */  \
  "sqxtn  v2.8b, v14.8h\n"        /*  20, cvt int16 to int8 */  \
  "sqxtn  v3.8b, v16.8h\n"        /*  30, cvt int16 to int8 */  \
  "sqxtn  v4.8b, v18.8h\n"        /*  40, cvt int16 to int8 */  \
  "sqxtn  v5.8b, v20.8h\n"        /*  50, cvt int16 to int8 */  \
  "sqxtn  v6.8b, v22.8h\n"        /*  60, cvt int16 to int8 */  \
  "sqxtn  v7.8b, v24.8h\n"        /*  70, cvt int16 to int8 */  \
  "st1 {v0.8b},[%[c_ptr0]], #8\n" /* store r0 */                \
  "st1 {v1.8b},[%[c_ptr1]], #8\n" /* store r1 */                \
  "st1 {v2.8b},[%[c_ptr2]], #8\n" /* store r2 */                \
  "st1 {v3.8b},[%[c_ptr3]], #8\n" /* store r3 */                \
  "st1 {v4.8b},[%[c_ptr4]], #8\n" /* store r4 */                \
  "st1 {v5.8b},[%[c_ptr5]], #8\n" /* store r5 */                \
  "st1 {v6.8b},[%[c_ptr6]], #8\n" /* store r6 */                \
  "st1 {v7.8b},[%[c_ptr7]], #8\n" /* store r7 */

#define GEMM_SDOT_INT8_OUT                                         \
  GEMM_SDOT_CVT_INT32_TO_FP32                                      \
  GEMM_SDOT_RELU                                                   \
  GEMM_SDOT_RELU6                                                  \
  GEMM_SDOT_LEAKY_RELU                                             \
  GEMM_SDOT_HARD_SWISH                                             \
  "ld1  {v6.4s}, [%[vmax]]\n"     /* v8 = -127.f     */            \
  /* data >= -127 */                                               \
  "fcmge v0.4s, v8.4s, v6.4s\n"                                   \
  "fcmge v1.4s, v9.4s, v6.4s\n"                                   \
  "fcmge v2.4s, v10.4s, v6.4s\n"                                   \
  "fcmge v3.4s, v11.4s, v6.4s\n"                                   \
  "fcmge v4.4s, v12.4s, v6.4s\n"                                   \
  "fcmge v5.4s, v13.4s, v6.4s\n"                                   \
  "fcmge v7.4s, v14.4s, v6.4s\n"                                   \
  /* choose data */                                                \
  "bif v8.16b, v6.16b, v0.16b\n"                                  \
  "fcmge v0.4s, v15.4s, v6.4s\n"                                   \
  "bif v9.16b, v6.16b, v1.16b\n"                                  \
  "bif v10.16b, v6.16b, v2.16b\n"                                 \
  "bif v11.16b, v6.16b, v3.16b\n"                                  \
  "bif v12.16b, v6.16b, v4.16b\n"                                  \
  "bif v13.16b, v6.16b, v5.16b\n"                                  \
  "bif v14.16b, v6.16b, v7.16b\n"                                  \
  "bif v15.16b, v6.16b, v0.16b \n"                                 \
  "fcvtas v0.4s, v8.4s\n"         /*  00, cvt to int */         \
  "fcvtas v1.4s, v9.4s\n"         /*  01, cvt to int */         \
  "fcvtas v2.4s, v10.4s\n"        /*  02, cvt to int */         \
  "fcvtas v3.4s, v11.4s\n"        /*  10, cvt to int */         \
  "fcvtas v4.4s, v12.4s\n"        /*  11, cvt to int */         \
  "fcvtas v5.4s, v13.4s\n"        /*  12, cvt to int */         \
  "fcvtas v6.4s, v14.4s\n"        /*  20, cvt to int */         \
  "fcvtas v7.4s, v15.4s\n"        /*  21, cvt to int */         \
  "sqxtn  v10.4h, v0.4s\n"        /*  00, cvt int32 to int16 */ \
  "sqxtn2 v10.8h, v1.4s\n"        /*  01, cvt int32 to int16 */ \
  "sqxtn  v11.4h, v2.4s\n"        /*  02, cvt int32 to int16 */ \
  "sqxtn  v12.4h, v3.4s\n"        /*  10, cvt int32 to int16 */ \
  "sqxtn2 v12.8h, v4.4s\n"        /*  11, cvt int32 to int16 */ \
  "sqxtn  v13.4h, v5.4s\n"        /*  12, cvt int32 to int16 */ \
  "sqxtn  v14.4h, v6.4s\n"        /*  20, cvt int32 to int16 */ \
  "ld1  {v6.4s}, [%[vmax]]\n"     /* v8 = -127.f     */            \
  "sqxtn2 v14.8h, v7.4s\n"        /*  21, cvt int32 to int16 */ \
  /* data >= -127 */                                               \
  "fcmge v0.4s, v16.4s, v6.4s\n"                                   \
  "fcmge v1.4s, v17.4s, v6.4s\n"                                   \
  "fcmge v2.4s, v18.4s, v6.4s\n"                                   \
  "fcmge v3.4s, v19.4s, v6.4s\n"                                   \
  "fcmge v4.4s, v20.4s, v6.4s\n"                                   \
  "fcmge v5.4s, v21.4s, v6.4s\n"                                   \
  "fcmge v7.4s, v22.4s, v6.4s\n"                                   \
  "fcmge v8.4s, v23.4s, v6.4s\n"                                   \
  "fcmge v9.4s, v24.4s, v6.4s\n"                                   \
  /* choose data */                                                \
  "bif v16.16b, v6.16b, v0.16b\n"                                  \
  "fcmge v0.4s, v25.4s, v6.4s\n"                                   \
  "bif v17.16b, v6.16b, v1.16b\n"                                  \
  "bif v18.16b, v6.16b, v2.16b\n"                                  \
  "bif v19.16b, v6.16b, v3.16b\n"                                  \
  "bif v20.16b, v6.16b, v4.16b\n"                                  \
  "bif v21.16b, v6.16b, v5.16b\n"                                  \
  "bif v22.16b, v6.16b, v7.16b\n"                                  \
  "bif v23.16b, v6.16b, v8.16b\n"                                  \
  "bif v24.16b, v6.16b, v9.16b\n"                                  \
  "bif v25.16b, v6.16b, v0.16b\n"                                  \
  "fcvtas v0.4s, v16.4s\n"        /*  22, cvt to int */         \
  "fcvtas v1.4s, v17.4s\n"        /*  30, cvt to int */         \
  "fcvtas v2.4s, v18.4s\n"        /*  31, cvt to int */         \
  "fcvtas v3.4s, v19.4s\n"        /*  32, cvt to int */         \
  "fcvtas v4.4s, v20.4s\n"        /*  40, cvt to int */         \
  "fcvtas v5.4s, v21.4s\n"        /*  41, cvt to int */         \
  "fcvtas v6.4s, v22.4s\n"        /*  42, cvt to int */         \
  "fcvtas v7.4s, v23.4s\n"        /*  50, cvt to int */         \
  "fcvtas v8.4s, v24.4s\n"        /*  51, cvt to int */         \
  "fcvtas v9.4s, v25.4s\n"        /*  52, cvt to int */         \
  "sqxtn  v15.4h, v0.4s\n"        /*  22, cvt int32 to int16 */ \
  "sqxtn  v16.4h, v1.4s\n"        /*  30, cvt int32 to int16 */ \
  "sqxtn2 v16.8h, v2.4s\n"        /*  31, cvt int32 to int16 */ \
  "sqxtn  v17.4h, v3.4s\n"        /*  32, cvt int32 to int16 */ \
  "sqxtn  v18.4h, v4.4s\n"        /*  40, cvt int32 to int16 */ \
  "sqxtn2 v18.8h, v5.4s\n"        /*  41, cvt int32 to int16 */ \
  "sqxtn  v19.4h, v6.4s\n"        /*  42, cvt int32 to int16 */ \
  "sqxtn  v20.4h, v7.4s\n"        /*  50, cvt int32 to int16 */ \
  "sqxtn2 v20.8h, v8.4s\n"        /*  51, cvt int32 to int16 */ \
  "ld1  {v6.4s}, [%[vmax]]\n"     /* v8 = -127.f     */            \
  "sqxtn  v21.4h, v9.4s\n"        /*  52, cvt int32 to int16 */ \
  /* data >= -127 */                                               \
  "fcmge v0.4s, v26.4s, v6.4s\n"                                   \
  "fcmge v1.4s, v27.4s, v6.4s\n"                                   \
  "fcmge v2.4s, v28.4s, v6.4s\n"                                   \
  "fcmge v3.4s, v29.4s, v6.4s\n"                                   \
  "fcmge v4.4s, v30.4s, v6.4s\n"                                   \
  "fcmge v5.4s, v31.4s, v6.4s\n"                                   \
  /* choose data */                                                \
  "bif v26.16b, v6.16b, v0.16b\n"                                  \
  "bif v27.16b, v6.16b, v1.16b\n"                                  \
  "bif v28.16b, v6.16b, v2.16b\n"                                  \
  "bif v29.16b, v6.16b, v3.16b\n"                                  \
  "bif v30.16b, v6.16b, v4.16b\n"                                  \
  "bif v31.16b, v6.16b, v5.16b\n"                                  \
  "fcvtas v0.4s, v26.4s\n"        /*  60, cvt to int */         \
  "fcvtas v1.4s, v27.4s\n"        /*  61, cvt to int */         \
  "fcvtas v2.4s, v28.4s\n"        /*  62, cvt to int */         \
  "fcvtas v3.4s, v29.4s\n"        /*  70, cvt to int */         \
  "fcvtas v4.4s, v30.4s\n"        /*  71, cvt to int */         \
  "fcvtas v5.4s, v31.4s\n"        /*  72, cvt to int */         \
  "sqxtn  v22.4h, v0.4s\n"        /*  60, cvt int32 to int16 */ \
  "sqxtn2 v22.8h, v1.4s\n"        /*  61, cvt int32 to int16 */ \
  "sqxtn  v23.4h, v2.4s\n"        /*  62, cvt int32 to int16 */ \
  "sqxtn  v24.4h, v3.4s\n"        /*  70, cvt int32 to int16 */ \
  "sqxtn2 v24.8h, v4.4s\n"        /*  71, cvt int32 to int16 */ \
  "sqxtn  v25.4h, v5.4s\n"        /*  72, cvt int32 to int16 */ \
  "sqxtn  v0.8b, v10.8h\n"        /*  00, cvt int16 to int8 */  \
  "sqxtn  v1.8b, v12.8h\n"        /*  10, cvt int16 to int8 */  \
  "sqxtn  v2.8b, v14.8h\n"        /*  20, cvt int16 to int8 */  \
  "sqxtn  v3.8b, v16.8h\n"        /*  30, cvt int16 to int8 */  \
  "sqxtn  v4.8b, v18.8h\n"        /*  40, cvt int16 to int8 */  \
  "sqxtn  v5.8b, v20.8h\n"        /*  50, cvt int16 to int8 */  \
  "sqxtn  v6.8b, v22.8h\n"        /*  60, cvt int16 to int8 */  \
  "sqxtn  v7.8b, v24.8h\n"        /*  70, cvt int16 to int8 */  \
  "st1 {v0.8b},[%[c_ptr0]], #8\n" /* store r0 */                \
  "sqxtn  v8.8b, v11.8h\n"        /*  0, cvt int16 to int8 */   \
  "st1 {v1.8b},[%[c_ptr1]], #8\n" /* store r1 */                \
  "sqxtn  v9.8b, v13.8h\n"        /*  1, cvt int16 to int8 */   \
  "st1 {v2.8b},[%[c_ptr2]], #8\n" /* store r2 */                \
  "sqxtn  v10.8b, v15.8h\n"       /*  2, cvt int16 to int8 */   \
  "st1 {v3.8b},[%[c_ptr3]], #8\n" /* store r3 */                \
  "sqxtn  v11.8b, v17.8h\n"       /*  3, cvt int16 to int8 */   \
  "st1 {v4.8b},[%[c_ptr4]], #8\n" /* store r4 */                \
  "sqxtn  v12.8b, v19.8h\n"       /*  4, cvt int16 to int8 */   \
  "st1 {v5.8b},[%[c_ptr5]], #8\n" /* store r5 */                \
  "sqxtn  v13.8b, v21.8h\n"       /*  5, cvt int16 to int8 */   \
  "st1 {v6.8b},[%[c_ptr6]], #8\n" /* store r6 */                \
  "sqxtn  v14.8b, v23.8h\n"       /*  6, cvt int16 to int8 */   \
  "st1 {v7.8b},[%[c_ptr7]], #8\n" /* store r7 */                \
  "sqxtn  v15.8b, v25.8h\n"       /*  7, cvt int16 to int8 */   \
  "str s8,[%[c_ptr0]], #4\n"      /* store r0 */                \
  "str s9,[%[c_ptr1]], #4\n"      /* store r1 */                \
  "str s10,[%[c_ptr2]], #4\n"     /* store r2 */                \
  "str s11,[%[c_ptr3]], #4\n"     /* store r3 */                \
  "str s12,[%[c_ptr4]], #4\n"     /* store r4 */                \
  "str s13,[%[c_ptr5]], #4\n"     /* store r5 */                \
  "str s14,[%[c_ptr6]], #4\n"     /* store r6 */                \
  "str s15,[%[c_ptr7]], #4\n"     /* store r7 */

#define GEMM_SDOT_INT32_OUT_8x4                                 \
  "st1 {v8.4s},[%[c_ptr0]],  #16\n"   /* store r0 */            \
  "st1 {v11.4s},[%[c_ptr1]], #16\n" /* store r1 */              \
  "st1 {v14.4s},[%[c_ptr2]], #16\n" /* store r2 */              \
  "st1 {v17.4s},[%[c_ptr3]], #16\n" /* store r3 */              \
  "st1 {v20.4s},[%[c_ptr4]], #16\n" /* store r4 */              \
  "st1 {v23.4s},[%[c_ptr5]], #16\n" /* store r5 */              \
  "st1 {v26.4s},[%[c_ptr6]], #16\n" /* store r6 */              \
  "st1 {v29.4s},[%[c_ptr7]], #16\n" /* store r7 */

#define GEMM_SDOT_INT32_OUT_8x8                                 \
  "st1 {v8.4s, v9.4s},[%[c_ptr0]], #32\n"   /* store r0 */      \
  "st1 {v11.4s, v12.4s},[%[c_ptr1]], #32\n" /* store r1 */      \
  "st1 {v14.4s, v15.4s},[%[c_ptr2]], #32\n" /* store r2 */      \
  "st1 {v17.4s, v18.4s},[%[c_ptr3]], #32\n" /* store r3 */      \
  "st1 {v20.4s, v21.4s},[%[c_ptr4]], #32\n" /* store r4 */      \
  "st1 {v23.4s, v24.4s},[%[c_ptr5]], #32\n" /* store r5 */      \
  "st1 {v26.4s, v27.4s},[%[c_ptr6]], #32\n" /* store r6 */      \
  "st1 {v29.4s, v30.4s},[%[c_ptr7]], #32\n" /* store r7 */

#define GEMM_SDOT_INT32_OUT                                        \
  "st1 {v8.4s, v9.4s, v10.4s},[%[c_ptr0]], #48\n"   /* store r0 */ \
  "st1 {v11.4s, v12.4s, v13.4s},[%[c_ptr1]], #48\n" /* store r1 */ \
  "st1 {v14.4s, v15.4s, v16.4s},[%[c_ptr2]], #48\n" /* store r2 */ \
  "st1 {v17.4s, v18.4s, v19.4s},[%[c_ptr3]], #48\n" /* store r3 */ \
  "st1 {v20.4s, v21.4s, v22.4s},[%[c_ptr4]], #48\n" /* store r4 */ \
  "st1 {v23.4s, v24.4s, v25.4s},[%[c_ptr5]], #48\n" /* store r5 */ \
  "st1 {v26.4s, v27.4s, v28.4s},[%[c_ptr6]], #48\n" /* store r6 */ \
  "st1 {v29.4s, v30.4s, v31.4s},[%[c_ptr7]], #48\n" /* store r7 */
// clang-format on

template <>
inline void gemm_sdot_int8_kernel_8x4(const int8_t* a_ptr,
                                      const int8_t*& b_ptr,  // NOLINT
                                      const float* bias,
                                      float32_t*& c_ptr0,  // NOLINT
                                      float32_t*& c_ptr1,  // NOLINT
                                      float32_t*& c_ptr2,  // NOLINT
                                      float32_t*& c_ptr3,  // NOLINT
                                      float32_t*& c_ptr4,  // NOLINT
                                      float32_t*& c_ptr5,  // NOLINT
                                      float32_t*& c_ptr6,  // NOLINT
                                      float32_t*& c_ptr7,  // NOLINT
                                      const float32_t* scale,
                                      const float32_t* alpha,
                                      int is_relu,
                                      int k,
                                      int tail) {
  // clang-format off
  asm volatile(  GEMM_SDOT_INT8_KERNEL_8x4
                 GEMM_SDOT_FP32_OUT_8x4
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [tail] "+r"(tail),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5),
                 [c_ptr6] "+r"(c_ptr6),
                 [c_ptr7] "+r"(c_ptr7)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu),
                  [alpha] "r"(alpha)
               : "cc","memory","v0","v1","v2",
                 "v3","v4","v5","v6","v7","v8","v9","v10",
                 "v11","v12","v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22","v23","v24",
                 "v25","v26","v27","v28","v29","v30","v31");
  // clang-format on
}

template <>
inline void gemm_sdot_int8_kernel_8x8(const int8_t* a_ptr,
                                      const int8_t*& b_ptr,  // NOLINT
                                      const float* bias,
                                      float32_t*& c_ptr0,  // NOLINT
                                      float32_t*& c_ptr1,  // NOLINT
                                      float32_t*& c_ptr2,  // NOLINT
                                      float32_t*& c_ptr3,  // NOLINT
                                      float32_t*& c_ptr4,  // NOLINT
                                      float32_t*& c_ptr5,  // NOLINT
                                      float32_t*& c_ptr6,  // NOLINT
                                      float32_t*& c_ptr7,  // NOLINT
                                      const float32_t* scale,
                                      const float32_t* alpha,
                                      int is_relu,
                                      int k,
                                      int tail) {
  // clang-format off
  asm volatile(GEMM_SDOT_INT8_KERNEL_8x8
               GEMM_SDOT_FP32_OUT_8x8
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [tail] "+r"(tail),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5),
                 [c_ptr6] "+r"(c_ptr6),
                 [c_ptr7] "+r"(c_ptr7)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu),
                  [alpha] "r"(alpha)
               : "cc","memory","v0","v1","v2",
                 "v3","v4","v5","v6","v7","v8","v9","v10",
                 "v11","v12","v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22","v23","v24",
                 "v25","v26","v27","v28","v29","v30","v31");
  // clang-format on
}
template <>
inline void gemm_sdot_int8_kernel(const int8_t* a_ptr,
                                  const int8_t*& b_ptr,  // NOLINT
                                  const float* bias,
                                  float32_t*& c_ptr0,  // NOLINT
                                  float32_t*& c_ptr1,  // NOLINT
                                  float32_t*& c_ptr2,  // NOLINT
                                  float32_t*& c_ptr3,  // NOLINT
                                  float32_t*& c_ptr4,  // NOLINT
                                  float32_t*& c_ptr5,  // NOLINT
                                  float32_t*& c_ptr6,  // NOLINT
                                  float32_t*& c_ptr7,  // NOLINT
                                  const float32_t* scale,
                                  const float32_t* alpha,
                                  int is_relu,
                                  int k,
                                  int tail) {
  // clang-format off
  asm volatile(GEMM_SDOT_INT8_KERNEL
               GEMM_SDOT_FP32_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [tail] "+r"(tail),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5),
                 [c_ptr6] "+r"(c_ptr6),
                 [c_ptr7] "+r"(c_ptr7)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu),
                  [alpha] "r"(alpha)
               : "cc","memory","v0","v1","v2",
                 "v3","v4","v5","v6","v7","v8","v9","v10",
                 "v11","v12","v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22","v23","v24",
                 "v25","v26","v27","v28","v29","v30","v31");
  // clang-format on
}

template <>
inline void gemm_sdot_int8_kernel_8x4(const int8_t* a_ptr,
                                      const int8_t*& b_ptr,  // NOLINT
                                      const float* bias,
                                      int8_t*& c_ptr0,  // NOLINT
                                      int8_t*& c_ptr1,  // NOLINT
                                      int8_t*& c_ptr2,  // NOLINT
                                      int8_t*& c_ptr3,  // NOLINT
                                      int8_t*& c_ptr4,  // NOLINT
                                      int8_t*& c_ptr5,  // NOLINT
                                      int8_t*& c_ptr6,  // NOLINT
                                      int8_t*& c_ptr7,  // NOLINT
                                      const float32_t* scale,
                                      const float32_t* alpha,
                                      int is_relu,
                                      int k,
                                      int tail) {
  // clang-format off
  float32_t vmax[4] = {-127.0, -127.0, -127.0, -127.0};
  asm volatile(GEMM_SDOT_INT8_KERNEL_8x4 GEMM_SDOT_INT8_OUT_8x4
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [tail] "+r"(tail),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5),
                 [c_ptr6] "+r"(c_ptr6),
                 [c_ptr7] "+r"(c_ptr7)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu), [vmax] "r"(vmax),
                 [alpha] "r"(alpha)
               : "cc","memory","v0","v1","v2","v3",
                 "v4","v5","v6","v7","v8","v9","v10",
                 "v11","v12","v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22","v23","v24",
                 "v25","v26","v27","v28","v29","v30","v31");
  // clang-format on
}

template <>
inline void gemm_sdot_int8_kernel_8x8(const int8_t* a_ptr,
                                      const int8_t*& b_ptr,  // NOLINT
                                      const float* bias,
                                      int8_t*& c_ptr0,  // NOLINT
                                      int8_t*& c_ptr1,  // NOLINT
                                      int8_t*& c_ptr2,  // NOLINT
                                      int8_t*& c_ptr3,  // NOLINT
                                      int8_t*& c_ptr4,  // NOLINT
                                      int8_t*& c_ptr5,  // NOLINT
                                      int8_t*& c_ptr6,  // NOLINT
                                      int8_t*& c_ptr7,  // NOLINT
                                      const float32_t* scale,
                                      const float32_t* alpha,
                                      int is_relu,
                                      int k,
                                      int tail) {
  // clang-format off
  float32_t vmax[4] = {-127.0, -127.0, -127.0, -127.0};
  asm volatile(GEMM_SDOT_INT8_KERNEL_8x8 GEMM_SDOT_INT8_OUT_8x8
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [tail] "+r"(tail),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5),
                 [c_ptr6] "+r"(c_ptr6),
                 [c_ptr7] "+r"(c_ptr7)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu), [vmax] "r"(vmax),
                 [alpha] "r"(alpha)
               : "cc","memory","v0","v1","v2","v3",
                 "v4","v5","v6","v7","v8","v9","v10",
                 "v11","v12","v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22","v23","v24",
                 "v25","v26","v27","v28","v29","v30","v31");
  // clang-format on
}

template <>
inline void gemm_sdot_int8_kernel(const int8_t* a_ptr,
                                  const int8_t*& b_ptr,  // NOLINT
                                  const float* bias,
                                  int8_t*& c_ptr0,  // NOLINT
                                  int8_t*& c_ptr1,  // NOLINT
                                  int8_t*& c_ptr2,  // NOLINT
                                  int8_t*& c_ptr3,  // NOLINT
                                  int8_t*& c_ptr4,  // NOLINT
                                  int8_t*& c_ptr5,  // NOLINT
                                  int8_t*& c_ptr6,  // NOLINT
                                  int8_t*& c_ptr7,  // NOLINT
                                  const float32_t* scale,
                                  const float32_t* alpha,
                                  int is_relu,
                                  int k,
                                  int tail) {
  // clang-format off
  float32_t vmax[4] = {-127.0, -127.0, -127.0, -127.0};
  asm volatile(GEMM_SDOT_INT8_KERNEL GEMM_SDOT_INT8_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [tail] "+r"(tail),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5),
                 [c_ptr6] "+r"(c_ptr6),
                 [c_ptr7] "+r"(c_ptr7)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu), [vmax] "r"(vmax),
                 [alpha] "r"(alpha)
               : "cc","memory","v0","v1","v2","v3",
                 "v4","v5","v6","v7","v8","v9","v10",
                 "v11","v12","v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22","v23","v24",
                 "v25","v26","v27","v28","v29","v30","v31");
  // clang-format on
}

template <>
inline void gemm_sdot_int8_kernel_8x4(const int8_t* a_ptr,
                                      const int8_t*& b_ptr,  // NOLINT
                                      const float* bias,
                                      int32_t*& c_ptr0,  // NOLINT
                                      int32_t*& c_ptr1,  // NOLINT
                                      int32_t*& c_ptr2,  // NOLINT
                                      int32_t*& c_ptr3,  // NOLINT
                                      int32_t*& c_ptr4,  // NOLINT
                                      int32_t*& c_ptr5,  // NOLINT
                                      int32_t*& c_ptr6,  // NOLINT
                                      int32_t*& c_ptr7,  // NOLINT
                                      const float32_t* scale,
                                      const float32_t* alpha,
                                      int is_relu,
                                      int k,
                                      int tail) {
  // clang-format off
  float32_t vmax[4] = {-127.0, -127.0, -127.0, -127.0};
  asm volatile(GEMM_SDOT_INT8_KERNEL_8x4 GEMM_SDOT_INT32_OUT_8x4
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [tail] "+r"(tail),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5),
                 [c_ptr6] "+r"(c_ptr6),
                 [c_ptr7] "+r"(c_ptr7)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu), [vmax] "r"(vmax),
                 [alpha] "r"(alpha)
               : "cc","memory","v0","v1","v2","v3",
                 "v4","v5","v6","v7","v8","v9","v10",
                 "v11","v12","v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22","v23","v24",
                 "v25","v26","v27","v28","v29","v30","v31");
  // clang-format on
}

template <>
inline void gemm_sdot_int8_kernel_8x8(const int8_t* a_ptr,
                                      const int8_t*& b_ptr,  // NOLINT
                                      const float* bias,
                                      int32_t*& c_ptr0,  // NOLINT
                                      int32_t*& c_ptr1,  // NOLINT
                                      int32_t*& c_ptr2,  // NOLINT
                                      int32_t*& c_ptr3,  // NOLINT
                                      int32_t*& c_ptr4,  // NOLINT
                                      int32_t*& c_ptr5,  // NOLINT
                                      int32_t*& c_ptr6,  // NOLINT
                                      int32_t*& c_ptr7,  // NOLINT
                                      const float32_t* scale,
                                      const float32_t* alpha,
                                      int is_relu,
                                      int k,
                                      int tail) {
  // clang-format off
  float32_t vmax[4] = {-127.0, -127.0, -127.0, -127.0};
  asm volatile(GEMM_SDOT_INT8_KERNEL_8x8 GEMM_SDOT_INT32_OUT_8x8
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [tail] "+r"(tail),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5),
                 [c_ptr6] "+r"(c_ptr6),
                 [c_ptr7] "+r"(c_ptr7)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu), [vmax] "r"(vmax),
                 [alpha] "r"(alpha)
               : "cc","memory","v0","v1","v2","v3",
                 "v4","v5","v6","v7","v8","v9","v10",
                 "v11","v12","v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22","v23","v24",
                 "v25","v26","v27","v28","v29","v30","v31");
  // clang-format on
}

template <>
inline void gemm_sdot_int8_kernel(const int8_t* a_ptr,
                                  const int8_t*& b_ptr,  // NOLINT
                                  const float* bias,
                                  int32_t*& c_ptr0,  // NOLINT
                                  int32_t*& c_ptr1,  // NOLINT
                                  int32_t*& c_ptr2,  // NOLINT
                                  int32_t*& c_ptr3,  // NOLINT
                                  int32_t*& c_ptr4,  // NOLINT
                                  int32_t*& c_ptr5,  // NOLINT
                                  int32_t*& c_ptr6,  // NOLINT
                                  int32_t*& c_ptr7,  // NOLINT
                                  const float32_t* scale,
                                  const float32_t* alpha,
                                  int is_relu,
                                  int k,
                                  int tail) {
  // clang-format off
  float32_t vmax[4] = {-127.0, -127.0, -127.0, -127.0};
  asm volatile(GEMM_SDOT_INT8_KERNEL GEMM_SDOT_INT32_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [tail] "+r"(tail),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5),
                 [c_ptr6] "+r"(c_ptr6),
                 [c_ptr7] "+r"(c_ptr7)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu), [vmax] "r"(vmax),
                 [alpha] "r"(alpha)
               : "cc","memory","v0","v1","v2","v3",
                 "v4","v5","v6","v7","v8","v9","v10",
                 "v11","v12","v13","v14","v15","v16","v17",
                 "v18","v19","v20","v21","v22","v23","v24",
                 "v25","v26","v27","v28","v29","v30","v31");
  // clang-format on
}
#endif

#else  // armv7
#define GEMM_DOT_CVT_INT32_TO_FP32      \
  "vld1.32  {d0-d1}, [%[scale]]!    \n" \
  "vld1.32  {d2-d3}, [%[bias_ptr]]! \n" \
  "vcvt.f32.s32     q2, q4          \n" \
  "vcvt.f32.s32     q3, q5          \n" \
  "vdup.32    q4,   d2[0]           \n" \
  "vdup.32    q5,   d2[0]           \n" \
  "vmla.f32   q4,   q2, d0[0]       \n" \
  "vmla.f32   q5,   q3, d0[0]       \n" \
  "vcvt.f32.s32     q2, q6          \n" \
  "vcvt.f32.s32     q3, q7          \n" \
  "vdup.32    q6,   d2[1]           \n" \
  "vdup.32    q7,   d2[1]           \n" \
  "vmla.f32   q6,   q2, d0[1]       \n" \
  "vmla.f32   q7,   q3, d0[1]       \n" \
  "vcvt.f32.s32     q2, q8          \n" \
  "vcvt.f32.s32     q3, q9          \n" \
  "vdup.32    q8,   d3[0]           \n" \
  "vdup.32    q9,   d3[0]           \n" \
  "vmla.f32   q8,   q2, d1[0]       \n" \
  "vmla.f32   q9,   q3, d1[0]       \n" \
  "vcvt.f32.s32     q2, q10         \n" \
  "vcvt.f32.s32     q3, q11         \n" \
  "vdup.32    q10,  d3[1]           \n" \
  "vdup.32    q11,  d3[1]           \n" \
  "vmla.f32   q10,  q2, d1[1]       \n" \
  "vmla.f32   q11,  q3, d1[1]       \n" \
  "vld1.32  {d0}, [%[scale]]        \n" \
  "vld1.32  {d2}, [%[bias_ptr]]     \n" \
  "vcvt.f32.s32     q2, q12         \n" \
  "vcvt.f32.s32     q3, q13         \n" \
  "vdup.32    q12,  d2[0]           \n" \
  "vdup.32    q13,  d2[0]           \n" \
  "vmla.f32   q12,  q2, d0[0]       \n" \
  "vmla.f32   q13,  q3, d0[0]       \n" \
  "vcvt.f32.s32     q2, q14         \n" \
  "vcvt.f32.s32     q3, q15         \n" \
  "vdup.32    q14,  d2[1]           \n" \
  "vdup.32    q15,  d2[1]           \n" \
  "vmla.f32   q14,  q2, d0[1]       \n" \
  "vmla.f32   q15,  q3, d0[1]       \n"

#define GEMM_DOT_ST_FP32           \
  "vst1.I32 {q4}, [%[c_ptr0]]! \n" \
  "vst1.I32 {q6}, [%[c_ptr1]]! \n" \
  "vst1.I32 {q5}, [%[c_ptr0]]! \n" \
  "vst1.I32 {q7}, [%[c_ptr1]]! \n" \
  "vst1.I32 {q8}, [%[c_ptr2]]! \n" \
  "vst1.I32 {q9}, [%[c_ptr2]]! \n" \
  "vst1.I32 {q10},[%[c_ptr3]]! \n" \
  "vst1.I32 {q11},[%[c_ptr3]]! \n" \
  "vst1.I32 {q12},[%[c_ptr4]]! \n" \
  "vst1.I32 {q13},[%[c_ptr4]]! \n" \
  "vst1.I32 {q14},[%[c_ptr5]]! \n" \
  "vst1.I32 {q15},[%[c_ptr5]]! \n"

#define GEMM_DOT_RELU                            \
  "cmp    %[relu],   #0      \n" /* skip relu */ \
  "beq    12f                \n"                 \
  "cmp    %[relu],    #1     \n" /* skip relu */ \
  "vmov.f32   q0, #0.0       \n" /* for relu*/   \
  "bne    13f                \n" /* other act */ \
  "vmax.f32   q4,   q4,   q0 \n" /* relu*/       \
  "vmax.f32   q5,   q5,   q0 \n" /* relu*/       \
  "vmax.f32   q6,   q6,   q0 \n" /* relu*/       \
  "vmax.f32   q7,   q7,   q0 \n" /* relu*/       \
  "vmax.f32   q8,   q8,   q0 \n" /* relu*/       \
  "vmax.f32   q9,   q9,   q0 \n" /* relu*/       \
  "vmax.f32   q10,  q10,  q0 \n" /* relu*/       \
  "vmax.f32   q11,  q11,  q0 \n" /* relu*/       \
  "vmax.f32   q12,  q12,  q0 \n" /* relu*/       \
  "vmax.f32   q13,  q13,  q0 \n" /* relu*/       \
  "vmax.f32   q14,  q14,  q0 \n" /* relu*/       \
  "vmax.f32   q15,  q15,  q0 \n" /* relu*/       \
  "b      12f                \n" /* relu end */

#define GEMM_DOT_RELU6                      \
  "13:                       \n"            \
  "cmp    %[relu],   #2\n" /* skip relu6 */ \
  "bne   14f\n"                             \
  "vmax.f32   q4,   q4,   q0 \n" /* relu*/  \
  "vmax.f32   q5,   q5,   q0 \n" /* relu*/  \
  "vmax.f32   q6,   q6,   q0 \n" /* relu*/  \
  "vmax.f32   q7,   q7,   q0 \n" /* relu*/  \
  "vld1.32    {d2-d3}, [%[alpha]] \n"       \
  "vmax.f32   q8,   q8,   q0 \n" /* relu*/  \
  "vmax.f32   q9,   q9,   q0 \n" /* relu*/  \
  "vmax.f32   q10,  q10,  q0 \n" /* relu*/  \
  "vmax.f32   q11,  q11,  q0 \n" /* relu*/  \
  "vmax.f32   q12,  q12,  q0 \n" /* relu*/  \
  "vmax.f32   q13,  q13,  q0 \n" /* relu*/  \
  "vmax.f32   q14,  q14,  q0 \n" /* relu*/  \
  "vmax.f32   q15,  q15,  q0 \n" /* relu*/  \
  "vmin.f32   q4,   q4,   q1 \n" /* relu6*/ \
  "vmin.f32   q5,   q5,   q1 \n" /* relu6*/ \
  "vmin.f32   q6,   q6,   q1 \n" /* relu6*/ \
  "vmin.f32   q7,   q7,   q1 \n" /* relu6*/ \
  "vmin.f32   q8,   q8,   q1 \n" /* relu6*/ \
  "vmin.f32   q9,   q9,   q1 \n" /* relu6*/ \
  "vmin.f32   q10,  q10,  q1 \n" /* relu6*/ \
  "vmin.f32   q11,  q11,  q1 \n" /* relu6*/ \
  "vmin.f32   q12,  q12,  q1 \n" /* relu6*/ \
  "vmin.f32   q13,  q13,  q1 \n" /* relu6*/ \
  "vmin.f32   q14,  q14,  q1 \n" /* relu6*/ \
  "vmin.f32   q15,  q15,  q1 \n" /* relu6*/ \
  "b      12f                \n" /* relu6 end */

#define GEMM_DOT_LEAKY_RELU                               \
  "14:                      \n"                           \
  "cmp    %[relu],   #3\n" /* skip leakyrelu */           \
  "bne   15f\n"                                           \
  "vld1.32  {d2-d3}, [%[alpha]] \n" /* leakyrelu alpha */ \
  "vcge.f32 q2,   q4,   q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q4,   q1  \n"     /* vmulq_f32 */       \
  "vbif     q4,   q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q5,   q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q5,   q1  \n"     /* vmulq_f32 */       \
  "vbif     q5,   q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q6,   q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q6,   q1  \n"     /* vmulq_f32 */       \
  "vbif     q6,   q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q7,   q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q7,   q1  \n"     /* vmulq_f32 */       \
  "vbif     q7,   q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q8,   q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q8,   q1  \n"     /* vmulq_f32 */       \
  "vbif     q8,   q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q9,   q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q9,   q1  \n"     /* vmulq_f32 */       \
  "vbif     q9,   q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q10,  q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q10,  q1  \n"     /* vmulq_f32 */       \
  "vbif     q10,  q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q11,  q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q11,  q1  \n"     /* vmulq_f32 */       \
  "vbif     q11,  q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q12,  q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q12,  q1  \n"     /* vmulq_f32 */       \
  "vbif     q12,  q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q13,  q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q13,  q1  \n"     /* vmulq_f32 */       \
  "vbif     q13,  q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q14,  q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q14,  q1  \n"     /* vmulq_f32 */       \
  "vbif     q14,  q3,   q2  \n"     /* choose*/           \
  "vcge.f32 q2,   q15,  q0  \n"     /* vcgeq_f32 */       \
  "vmul.f32 q3,   q15,  q1  \n"     /* vmulq_f32 */       \
  "vbif     q15,  q3,   q2  \n"     /* choose*/           \
  "b      12f               \n"

#define GEMM_DOT_HARD_SWISH                                 \
  "15:                      \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vld1.32  {d2-d3}, [%[alpha]] \n" /* leakyrelu alpha */   \
  "vadd.f32 q3,   q4,   q2  \n"                             \
  "vmul.f32 q4,   q4,   q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q4,   q4,   q3  \n"                             \
  "vadd.f32 q3,   q5,   q2  \n"                             \
  "vmul.f32 q5,   q5,   q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q5,   q5,   q3  \n"                             \
  "vadd.f32 q3,   q6,   q2  \n"                             \
  "vmul.f32 q6,   q6,   q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q6,   q6,   q3  \n"                             \
  "vadd.f32 q3,   q7,   q2  \n"                             \
  "vmul.f32 q7,   q7,   q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q7,   q7,   q3  \n"                             \
  "vadd.f32 q3,   q8,   q2  \n"                             \
  "vmul.f32 q8,   q8,   q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q8,   q8,   q3  \n"                             \
  "vadd.f32 q3,   q9,   q2  \n"                             \
  "vmul.f32 q9,   q9,   q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q9,   q9,   q3  \n"                             \
  "vadd.f32 q3,   q10,  q2  \n"                             \
  "vmul.f32 q10,  q10,  q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q10,  q10,  q3  \n"                             \
  "vadd.f32 q3,   q11,  q2  \n"                             \
  "vmul.f32 q11,  q11,  q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q11,  q11,  q3  \n"                             \
  "vadd.f32 q3,   q12,  q2  \n"                             \
  "vmul.f32 q12,  q12,  q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q12,  q12,  q3  \n"                             \
  "vadd.f32 q3,   q13,  q2  \n"                             \
  "vmul.f32 q13,  q13,  q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q13,  q13,  q3  \n"                             \
  "vadd.f32 q3,   q14,  q2  \n"                             \
  "vmul.f32 q14,  q14,  q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q14,  q14,  q3  \n"                             \
  "vadd.f32 q3,   q15,  q2  \n"                             \
  "vmul.f32 q15,  q15,  q1  \n"                             \
  "vldr       d4,  [%[alpha], #48]      @ load threshold\n" \
  "vldr       d5,  [%[alpha], #56]      @ load threshold\n" \
  "vmax.f32 q3,   q3,   q0  \n"                             \
  "vmin.f32 q3,   q3,   q2  \n"                             \
  "vldr       d4,  [%[alpha], #32]      @ load offset\n"    \
  "vldr       d5,  [%[alpha], #40]      @ load offset\n"    \
  "vmul.f32 q15,  q15,  q3  \n"                             \
  "12:                      \n"

#define GEMM_DOT_ST_INT8                                     \
  "add %[alpha],    #16             \n"                      \
  "vld1.32    {d0-d1},    [%[alpha]]\n"                      \
  "vmov.f32   q1,   #0.5            \n"                      \
  "vmov.f32   q2,   #-0.5           \n"                      \
  "vcgt.f32   q3,   q4,   #0        \n"                      \
  "vbif.f32   q1,   q2,   q3        \n"                      \
  "vadd.f32   q4,   q1,   q4        \n"                      \
  "vmov.f32   q1,   #0.5            \n"                      \
  "vcgt.f32   q3,   q5,   #0        \n"                      \
  "vbif.f32   q1,   q2,   q3        \n"                      \
  "vadd.f32   q5,   q1,   q5        \n" /* data >= -127 */   \
  "vcge.f32   q1,   q4,   q0        \n"                      \
  "vcge.f32   q2,   q5,   q0        \n"                      \
  "vbif q4,   q0,   q1              \n"                      \
  "vbif q5,   q0,   q2              \n" /* fp32 to int32 */  \
  "vcvt.s32.f32     q1,   q4        \n"                      \
  "vcvt.s32.f32     q2,   q5        \n" /* int32 to int16 */ \
  "vqmovn.s32 d8,   q1              \n"                      \
  "vqmovn.s32 d9,   q2              \n" /* int16 to int8 */  \
  "vqmovn.s16 d2,   q4              \n"                      \
  "vst1.32    {d2}, [%[c_ptr0]]!    \n"                      \
                                                             \
  "vmov.f32   q1,   #0.5            \n"                      \
  "vmov.f32   q3,   #0.5            \n"                      \
  "vmov.f32   q2,   #-0.5           \n"                      \
  "vcgt.f32   q4,   q6,   #0        \n"                      \
  "vcgt.f32   q5,   q7,   #0        \n"                      \
  "vbif.f32   q1,   q2,   q4        \n"                      \
  "vbif.f32   q3,   q2,   q5        \n"                      \
  "vmov.f32   q4,   #0.5            \n"                      \
  "vmov.f32   q5,   #0.5            \n"                      \
  "vadd.f32   q6,   q1,   q6        \n"                      \
  "vadd.f32   q7,   q3,   q7        \n"                      \
  "vcgt.f32   q5,   q8,   #0        \n"                      \
  "vbif.f32   q4,   q2,   q5        \n"                      \
  "vadd.f32   q8,   q4,   q8        \n"                      \
  "vcgt.f32   q5,   q9,   #0        \n" /* data >= -127 */   \
  "vcge.f32   q1,   q6,   q0        \n"                      \
  "vcge.f32   q3,   q7,   q0        \n"                      \
  "vcge.f32   q4,   q8,   q0        \n"                      \
  "vbif q6,   q0,   q1              \n"                      \
  "vbif q7,   q0,   q3              \n"                      \
  "vbif q8,   q0,   q4              \n" /* fp32 to int32 */  \
  "vcvt.s32.f32     q1,   q6        \n"                      \
  "vcvt.s32.f32     q3,   q7        \n"                      \
  "vcvt.s32.f32     q4,   q8        \n" /* int32 to int16 */ \
  "vqmovn.s32 d12,  q1              \n"                      \
  "vqmovn.s32 d13,  q3              \n"                      \
  "vqmovn.s32 d16,  q4              \n"                      \
  "vmov.f32   q7,   #0.5            \n"                      \
  "vbif.f32   q7,   q2,   q5        \n"                      \
  "vadd.f32   q9,   q7,   q9        \n"                      \
  "vcge.f32   q5,   q9,   q0        \n"                      \
  "vbif q9,   q0,   q5              \n"                      \
  "vcvt.s32.f32     q5,   q9        \n"                      \
  "vqmovn.s32 d17,  q5              \n" /* int16 to int8 */  \
  "vqmovn.s16 d19,  q8              \n"                      \
  "vqmovn.s16 d18,  q6              \n"                      \
  "vst1.32    {d18},[%[c_ptr1]]!    \n"                      \
  "vst1.32    {d19},[%[c_ptr2]]!    \n"                      \
                                                             \
  "vmov.f32   q2,   #-0.5           \n"                      \
  "vmov.f32   q1,   #0.5            \n"                      \
  "vmov.f32   q3,   #0.5            \n"                      \
  "vmov.f32   q4,   #0.5            \n"                      \
  "vmov.f32   q5,   #0.5            \n"                      \
  "vcgt.f32   q6,   q10,  #0        \n"                      \
  "vcgt.f32   q7,   q11,  #0        \n"                      \
  "vcgt.f32   q8,   q12,  #0        \n"                      \
  "vcgt.f32   q9,   q13,  #0        \n"                      \
  "vbif.f32   q1,   q2,   q6        \n"                      \
  "vbif.f32   q3,   q2,   q7        \n"                      \
  "vbif.f32   q4,   q2,   q8        \n"                      \
  "vbif.f32   q5,   q2,   q9        \n"                      \
  "vmov.f32   q6,   #0.5            \n"                      \
  "vmov.f32   q7,   #0.5            \n"                      \
  "vcgt.f32   q8,   q14,  #0        \n"                      \
  "vcgt.f32   q9,   q15,  #0        \n"                      \
  "vbif.f32   q6,   q2,   q8        \n"                      \
  "vbif.f32   q7,   q2,   q9        \n"                      \
  "vadd.f32   q10,  q1,   q10       \n"                      \
  "vadd.f32   q11,  q3,   q11       \n"                      \
  "vadd.f32   q12,  q4,   q12       \n"                      \
  "vadd.f32   q13,  q5,   q13       \n"                      \
  "vadd.f32   q14,  q6,   q14       \n"                      \
  "vadd.f32   q15,  q7,   q15       \n"                      \
                                                             \
  "vcge.f32   q1,   q10,  q0        \n"                      \
  "vcge.f32   q3,   q11,  q0        \n"                      \
  "vcge.f32   q4,   q12,  q0        \n"                      \
  "vcge.f32   q5,   q13,  q0        \n"                      \
  "vcge.f32   q6,   q14,  q0        \n"                      \
  "vcge.f32   q7,   q15,  q0        \n"                      \
  "vbif       q10,  q0,   q1        \n"                      \
  "vbif       q11,  q0,   q3        \n"                      \
  "vbif       q12,  q0,   q4        \n"                      \
  "vbif       q13,  q0,   q5        \n"                      \
  "vbif       q14,  q0,   q6        \n"                      \
  "vbif       q15,  q0,   q7        \n" /* fp32 to int32 */  \
  "vcvt.s32.f32     q1,   q10       \n"                      \
  "vcvt.s32.f32     q3,   q11       \n"                      \
  "vcvt.s32.f32     q4,   q12       \n"                      \
  "vcvt.s32.f32     q5,   q13       \n"                      \
  "vcvt.s32.f32     q6,   q14       \n"                      \
  "vcvt.s32.f32     q7,   q15       \n" /* int32 to int16 */ \
  "vqmovn.s32 d16,  q1              \n"                      \
  "vqmovn.s32 d17,  q3              \n"                      \
  "vqmovn.s32 d18,  q4              \n"                      \
  "vqmovn.s32 d19,  q5              \n"                      \
  "vqmovn.s32 d20,  q6              \n"                      \
  "vqmovn.s32 d21,  q7              \n" /* int16 to int8 */  \
  "vqmovn.s16 d2,   q8              \n"                      \
  "vqmovn.s16 d3,   q9              \n"                      \
  "vqmovn.s16 d4,   q10             \n"                      \
  "sub %[alpha], #16                \n"                      \
  "vst1.32    {d2}, [%[c_ptr3]]!    \n"                      \
  "vst1.32    {d3}, [%[c_ptr4]]!    \n"                      \
  "vst1.32    {d4}, [%[c_ptr5]]!    \n"

#define GEMM_DOT_FP32_OUT    \
  GEMM_DOT_CVT_INT32_TO_FP32 \
  GEMM_DOT_RELU              \
  GEMM_DOT_RELU6             \
  GEMM_DOT_LEAKY_RELU        \
  GEMM_DOT_HARD_SWISH        \
  GEMM_DOT_ST_FP32

#define GEMM_DOT_INT8_OUT    \
  GEMM_DOT_CVT_INT32_TO_FP32 \
  GEMM_DOT_RELU              \
  GEMM_DOT_RELU6             \
  GEMM_DOT_LEAKY_RELU        \
  GEMM_DOT_HARD_SWISH        \
  GEMM_DOT_ST_INT8
#define GEMM_DOT_INT32_OUT        \
  "vst1.32 {q4}, [%[c_ptr0]]! \n" \
  "vst1.32 {q6}, [%[c_ptr1]]! \n" \
  "vst1.32 {q5}, [%[c_ptr0]]! \n" \
  "vst1.32 {q7}, [%[c_ptr1]]! \n" \
  "vst1.32 {q8}, [%[c_ptr2]]! \n" \
  "vst1.32 {q9}, [%[c_ptr2]]! \n" \
  "vst1.32 {q10},[%[c_ptr3]]! \n" \
  "vst1.32 {q11},[%[c_ptr3]]! \n" \
  "vst1.32 {q12},[%[c_ptr4]]! \n" \
  "vst1.32 {q13},[%[c_ptr4]]! \n" \
  "vst1.32 {q14},[%[c_ptr5]]! \n" \
  "vst1.32 {q15},[%[c_ptr5]]! \n"
template <typename Dtype>
inline void gemm_dot_int8_kernel(const int8_t* a_ptr,
                                 const int8_t*& b_ptr,  // NOLINT
                                 const float* bias,
                                 Dtype*& c_ptr0,  // NOLINT
                                 Dtype*& c_ptr1,  // NOLINT
                                 Dtype*& c_ptr2,  // NOLINT
                                 Dtype*& c_ptr3,  // NOLINT
                                 Dtype*& c_ptr4,  // NOLINT
                                 Dtype*& c_ptr5,  // NOLINT
                                 const float32_t* scale,
                                 const float32_t* alpha,
                                 int is_relu,
                                 int k,
                                 int rem);

template <>
inline void gemm_dot_int8_kernel(const int8_t* a_ptr,
                                 const int8_t*& b_ptr,  // NOLINT
                                 const float* bias,
                                 float32_t*& c_ptr0,  // NOLINT
                                 float32_t*& c_ptr1,  // NOLINT
                                 float32_t*& c_ptr2,  // NOLINT
                                 float32_t*& c_ptr3,  // NOLINT
                                 float32_t*& c_ptr4,  // NOLINT
                                 float32_t*& c_ptr5,  // NOLINT
                                 const float32_t* scale,
                                 const float32_t* alpha,
                                 int is_relu,
                                 int k,
                                 int tail) {
  float new_ptr[16] = {alpha[0],
                       alpha[1],
                       alpha[2],
                       alpha[3],
                       -127.0,
                       -127.0,
                       -127.0,
                       -127.0,
                       alpha[4],
                       alpha[5],
                       alpha[6],
                       alpha[7],
                       alpha[8],
                       alpha[9],
                       alpha[10],
                       alpha[11]};
  asm volatile(GEMM_DOT_INT8_KERNEL GEMM_DOT_FP32_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+&r"(b_ptr),
                 [k] "+r"(k),
                 [c_ptr0] "+&r"(c_ptr0),
                 [c_ptr1] "+&r"(c_ptr1),
                 [c_ptr2] "+&r"(c_ptr2),
                 [c_ptr3] "+&r"(c_ptr3),
                 [c_ptr4] "+&r"(c_ptr4),
                 [c_ptr5] "+&r"(c_ptr5)
               : [bias_ptr] "r"(bias),
                 [scale] "r"(scale),
                 [relu] "r"(is_relu),
                 [alpha] "r"(new_ptr)
               : "q0",
                 "q1",
                 "q2",
                 "q3",
                 "q4",
                 "q5",
                 "q6",
                 "q7",
                 "q8",
                 "q9",
                 "q10",
                 "q11",
                 "q12",
                 "q13",
                 "q14",
                 "q15",
                 "cc",
                 "memory");
  // clang-format on
}
template <>
inline void gemm_dot_int8_kernel(const int8_t* a_ptr,
                                 const int8_t*& b_ptr,  // NOLINT
                                 const float* bias,
                                 int8_t*& c_ptr0,  // NOLINT
                                 int8_t*& c_ptr1,  // NOLINT
                                 int8_t*& c_ptr2,  // NOLINT
                                 int8_t*& c_ptr3,  // NOLINT
                                 int8_t*& c_ptr4,  // NOLINT
                                 int8_t*& c_ptr5,  // NOLINT
                                 const float32_t* scale,
                                 const float32_t* alpha,
                                 int is_relu,
                                 int k,
                                 int tail) {
  float new_ptr[16] = {alpha[0],
                       alpha[1],
                       alpha[2],
                       alpha[3],
                       -127.0,
                       -127.0,
                       -127.0,
                       -127.0,
                       alpha[4],
                       alpha[5],
                       alpha[6],
                       alpha[7],
                       alpha[8],
                       alpha[9],
                       alpha[10],
                       alpha[11]};

  // clang-format off
  asm volatile(GEMM_DOT_INT8_KERNEL   GEMM_DOT_INT8_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu), 
                 [alpha] "r"(new_ptr)
               : "q0","q1","q2",
                 "q3","q4","q5","q6","q7","q8","q9","q10",
                 "q11","q12","q13","q14","q15","cc","memory");

  // clang-format on
}
template <>
inline void gemm_dot_int8_kernel(const int8_t* a_ptr,
                                 const int8_t*& b_ptr,  // NOLINT
                                 const float* bias,
                                 int32_t*& c_ptr0,  // NOLINT
                                 int32_t*& c_ptr1,  // NOLINT
                                 int32_t*& c_ptr2,  // NOLINT
                                 int32_t*& c_ptr3,  // NOLINT
                                 int32_t*& c_ptr4,  // NOLINT
                                 int32_t*& c_ptr5,  // NOLINT
                                 const float32_t* scale,
                                 const float32_t* alpha,
                                 int is_relu,
                                 int k,
                                 int tail) {
  float new_ptr[16] = {alpha[0],
                       alpha[1],
                       alpha[2],
                       alpha[3],
                       -127.0,
                       -127.0,
                       -127.0,
                       -127.0,
                       alpha[4],
                       alpha[5],
                       alpha[6],
                       alpha[7],
                       alpha[8],
                       alpha[9],
                       alpha[10],
                       alpha[11]};

  // clang-format off
  asm volatile(GEMM_DOT_INT8_KERNEL   GEMM_DOT_INT32_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [k] "+r"(k),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [c_ptr4] "+r"(c_ptr4),
                 [c_ptr5] "+r"(c_ptr5)
               : [bias_ptr] "r"(bias), [scale] "r"(scale), [relu] "r"(is_relu), 
                 [alpha] "r"(new_ptr)
               : "q0","q1","q2",
                 "q3","q4","q5","q6","q7","q8","q9","q10",
                 "q11","q12","q13","q14","q15","cc","memory");

  // clang-format on
}

// clang-format off
#define GEMM_INT8_KERNEL                                                  \
  "vld1.8 {d0-d1}, [%[a_ptr]: 128]!\n" /* load 4x2x2 int8, A, k2x2 */     \
  "vld1.8 {d4-d7}, [%[b_ptr]: 128]!\n" /* load 8x2x2 int8, B, k2x2 */     \
  "pld [%[a_ptr]]\n"                   /* preload A */                    \
  "veor   q8, q4, q4\n"                /* set bias to out00 */            \
  "veor   q9, q4, q4\n"                /* set bias to out01 */            \
  "pld [%[b_ptr]]\n"                   /* preload B */                    \
  "veor  q10, q5, q5\n"                /* set bias to out10 */            \
  "veor  q11, q5, q5\n"                /* set bias to out11 */            \
  "pld [%[b_ptr], #64]\n"              /* preload B */                    \
  "veor  q12, q6, q6\n"                /* set bias to out20 */            \
  "veor  q13, q6, q6\n"                /* set bias to out21 */            \
  "pld [%[b_ptr], #128]\n"             /* preload B */                    \
  "veor  q14, q7, q7\n"                /* set bias to out30 */            \
  "veor  q15, q7, q7\n"                /* set bias to out31 */            \
  "pld [%[a_ptr], #64]\n"              /* preload A */                    \
  "vext.8 d2, d0, d0, #2\n"            /* shift left circular by 2byte */ \
  "vext.8 d3, d1, d1, #2\n"            /* shift left circular by 2byte */ \
  "pld [%[b_ptr], #192]\n"             /* preload b */                    \
  "pld [%[b_ptr], #256]\n"             /* preload b */                    \
  "pld [%[a_ptr], #128]\n"             /* preload a */                    \
  "cmp    %[k],   #0\n"                /* check main loop count */        \
  "beq    3f\n"                        /* if k = 0, jump to remains */    \
  /* 1st r0, r1 */                                                        \
  "vmull.s8  q4, d0, d4\n"             /* a0 * b0 = c00 */                \
  "vmull.s8  q5, d0, d5\n"             /* a0 * b1 = c01 */                \
  "vmull.s8  q6, d2, d4\n"             /* a1 * b0 = c10 */                \
  "vmull.s8  q7, d2, d5\n"             /* a1 * b1 = c11 */                \
  "subs %[k], %[k], #1\n"              /* loop count -1 */                \
  /* 2nd r0, r1 */                                                        \
  "vmlal.s8  q4, d1, d6\n"             /* a0 * b0 = c00 */                \
  "vmlal.s8  q5, d1, d7\n"             /* a0 * b1 = c01 */                \
  "vrev64.32  q0, q0\n"                /* shift left circular by 4byte */ \
  "vmlal.s8  q6, d3, d6\n"             /* a1 * b0 = c10 */                \
  "vmlal.s8  q7, d3, d7\n"             /* a1 * b1 = c11 */                \
  "vrev64.32  q1, q1\n"                /* shift left circular by 4byte */ \
  "beq    8f\n"                        /* skip main loop */               \
  /* main loop*/                                                          \
  "0:\n"                               /* main loop */                    \
  /* 1st r2, r3 */                                                        \
  "vpadal.s16 q8, q4\n"                /* pair add and accumulate, c00 */ \
  "vmull.s8  q4, d0, d4\n"             /* a2 * b0 = c20 */                \
  "vpadal.s16 q9, q5\n"                /* pair add and accumulate, c01 */ \
  "vmull.s8  q5, d0, d5\n"             /* a2 * b1 = c21 */                \
  "vpadal.s16 q10,q6\n"                /* pair add and accumulate, c10 */ \
  "vmull.s8  q6, d2, d4\n"             /* a3 * b0 = c30 */                \
  "vpadal.s16 q11,q7\n"                /* pair add and accumulate, c11 */ \
  "vmull.s8  q7, d2, d5\n"             /* a3 * b1 = c31 */                \
  "vld1.8 {d4-d5}, [%[b_ptr]: 128]!\n" /* load 4x2x2 int8, B, k2x2 */     \
  /* 2nd r2, r3 */                                                        \
  "vmlal.s8  q4, d1, d6\n"             /* a0 * b0 = c00 */                \
  "vmlal.s8  q5, d1, d7\n"             /* a0 * b1 = c01 */                \
  "vld1.8 {d0-d1}, [%[a_ptr]: 128]!\n" /* load 4x2x2 int8, A, k2x2 */     \
  "vmlal.s8  q6, d3, d6\n"             /* a1 * b0 = c10 */                \
  "vmlal.s8  q7, d3, d7\n"             /* a1 * b1 = c11 */                \
  "vld1.8 {d6-d7}, [%[b_ptr]: 128]!\n" /* load 4x2x2 int8, B, k2x2 */     \
  /* pre process A */                                                     \
  "vext.8 d2, d0, d0, #2\n"            /* shift left circular by 2byte */ \
  "vext.8 d3, d1, d1, #2\n"            /* shift left circular by 2byte */ \
  /* 1st r0, r1 */                                                        \
  "vpadal.s16 q12,q4\n"                /* pair add and accumulate, c20 */ \
  "vmull.s8  q4, d0, d4\n"             /* a0 * b0 = c00 */                \
  "vpadal.s16 q13,q5\n"                /* pair add and accumulate, c21 */ \
  "vmull.s8  q5, d0, d5\n"             /* a0 * b1 = c01 */                \
  "vpadal.s16 q14,q6\n"                /* pair add and accumulate, c30 */ \
  "vmull.s8  q6, d2, d4\n"             /* a1 * b0 = c10 */                \
  "vpadal.s16 q15,q7\n"                /* pair add and accumulate, c31 */ \
  "vmull.s8  q7, d2, d5\n"             /* a1 * b1 = c11 */                \
  "subs %[k], %[k], #1\n"              /* loop count -1 */                \
  /* 2nd r0, r1 */                                                        \
  "vmlal.s8  q4, d1, d6\n"             /* a0 * b0 = c00 */                \
  "vmlal.s8  q5, d1, d7\n"             /* a0 * b1 = c01 */                \
  "vrev64.32  q0, q0\n"                /* shift left circular by 2 */     \
  "vmlal.s8  q6, d3, d6\n"             /* a1 * b0 = c10 */                \
  "vmlal.s8  q7, d3, d7\n"             /* a1 * b1 = c11 */                \
  "vrev64.32  q1, q1\n"                /* shift left circular by 2 */     \
  "bgt    0b\n"                        /* jump to main loop */            \
  "8:\n"                               /* end of main loop */             \
  /* 1st r2, r3 */                                                        \
  "vpadal.s16 q8, q4\n"                /* pair add and accumulate, c00 */ \
  "vmull.s8  q4, d0, d4\n"             /* a2 * b0 = c20 */                \
  "vpadal.s16 q9, q5\n"                /* pair add and accumulate, c01 */ \
  "vmull.s8  q5, d0, d5\n"             /* a2 * b1 = c21 */                \
  "vpadal.s16 q10,q6\n"                /* pair add and accumulate, c10 */ \
  "vmull.s8  q6, d2, d4\n"             /* a3 * b0 = c30 */                \
  "vpadal.s16 q11,q7\n"                /* pair add and accumulate, c11 */ \
  "vmull.s8  q7, d2, d5\n"             /* a3 * b1 = c31 */                \
  /* 2nd r2, r3 */                                                        \
  "vmlal.s8  q4, d1, d6\n"             /* a0 * b0 = c20 */                \
  "vmlal.s8  q5, d1, d7\n"             /* a0 * b1 = c21 */                \
  "vmlal.s8  q6, d3, d6\n"             /* a1 * b0 = c30 */                \
  "vmlal.s8  q7, d3, d7\n"             /* a1 * b1 = c31 */                \
  "cmp    %[rem],    #0\n"             /* skip remain */                  \
  "beq    5f\n"                                                           \
  "mov %[k],    #32\n"                   /* address offset */               \
  "vld1.8 {d0}, [%[a_ptr]]\n"          /* load a to d0, final */          \
  "vld1.8 {d4-d5}, [%[b_ptr]], %[k]\n"   /* load b to d4, d5 */             \
  "5:\n"                               /* skip rem */                     \
  "vpadal.s16 q12, q4\n"               /* pair add and accumulate, c20 */ \
  "vpadal.s16 q13, q5\n"               /* pair add and accumulate, c21 */ \
  "vpadal.s16 q14, q6\n"               /* pair add and accumulate, c30 */ \
  "vpadal.s16 q15, q7\n"               /* pair add and accumulate, c31 */ \
  /* process remain k */                                                  \
  "3:\n"                               /* process remain k */             \
  "cmp    %[rem],    #0\n"             /* skip remain */                  \
  "beq    7f\n"                                                           \
  /* process remain k */                                                  \
  "vext.8 d1, d0, d0, #2\n"            /* shift left 2bytes */            \
  "vext.8 d2, d0, d0, #4\n"            /* shift left 4bytes */            \
  "vext.8 d3, d0, d0, #6\n"            /* shift left 6bytes */            \
  /* 1st r0, r1 */                                                        \
  "vmull.s8  q4, d0, d4\n"             /* a0 * b0 = c00 */                \
  "vmull.s8  q5, d0, d5\n"             /* a0 * b1 = c01 */                \
  "vmull.s8  q6, d1, d4\n"             /* a1 * b0 = c10 */                \
  "vmull.s8  q7, d1, d5\n"             /* a1 * b1 = c11 */                \
  /* 1st r2, r3 */                                                        \
  "vpadal.s16 q8, q4\n"                /* pair add and accumulate, c00 */ \
  "vmull.s8  q4, d2, d4\n"             /* a2 * b0 = c20 */                \
  "vpadal.s16 q9, q5\n"                /* pair add and accumulate, c01 */ \
  "vmull.s8  q5, d2, d5\n"             /* a2 * b1 = c21 */                \
  "vpadal.s16 q10,q6\n"                /* pair add and accumulate, c10 */ \
  "vmull.s8  q6, d3, d4\n"             /* a3 * b0 = c30 */                \
  "vpadal.s16 q11,q7\n"                /* pair add and accumulate, c11 */ \
  "vmull.s8  q7, d3, d5\n"             /* a3 * b1 = c31 */                \
  "vpadal.s16 q12, q4\n"               /* pair add and accumulate, c20 */ \
  "vpadal.s16 q13, q5\n"               /* pair add and accumulate, c21 */ \
  "vpadal.s16 q14, q6\n"               /* pair add and accumulate, c30 */ \
  "vpadal.s16 q15, q7\n"               /* pair add and accumulate, c31 */ \
  "7: \n"                              /* do relu */                      \
  /* unpack the result */                                                 \
  /* trans 1 */                                                           \
  "vtrn.32    q8, q10\n" /* get q8:  a0b0, a1b0, a2b2, a3b2;*/            \
  /* q10: a1b1, a2b1, a3b3, a0b3 */                                       \
  "vtrn.32   q12, q14\n" /* get q12: a2b0, a3b0, a0b2, a1b2;*/            \
  /* q14: a3b1, a0b1, a1b3, a2b3 */                                       \
  "vtrn.32    q9, q11\n" /* get q9:  a0b0, a1b0, a2b2, a3b2;*/            \
  /* q11: a1b1, a2b1, a3b3, a0b3 */                                       \
  "vtrn.32   q13, q15\n" /* get q13: a2b0, a3b0, a0b2, a1b2;*/            \
  /* q15: a3b1, a0b1, a1b3, a2b3 */                                       \
  /* trans 2 */                                                           \
  "vswp   d17,    d24\n" /* get q8:  a0b0, a1b0, a2b0, a3b0;*/            \
  /* q12: a2b2, a3b2, a0b2, a1b2 */                                       \
  "vswp   d21,    d28\n" /* get q10: a1b1, a2b1, a3b1, a0b1;*/            \
  /* q14: a3b3, a0b3, a1b3, a2b3 */                                       \
  "vswp   d19,    d26\n" /* get q9:  a0b0, a1b0, a2b0, a3b0;*/            \
  /* q13: a2b2, a3b2, a0b2, a1b2 */                                       \
  "vswp   d23,    d30\n" /* get q11: a1b1, a2b1, a3b1, a0b1;*/            \
  /* q15: a3b3, a0b3, a1b3, a2b3 */                                       \
  /* shift */                                                             \
  "vext.8 q0, q10, q10, #12\n" /* circular shift left 1 */                \
  /* q0: a0b1, a1b1, a2b1, a3b1 */                                        \
  "vext.8 q2, q12, q12, #8\n"  /* circular shift left 2 */                \
  /* q2: a0b2, a1b2, a2b2, a3b2 */                                        \
  "vext.8 q4, q14, q14, #4\n"  /* circular shift left 3 */                \
  /* q4: a0b3, a1b3, a2b3, a3b3 */                                        \
  "vext.8 q1, q11, q11, #12\n" /* circular shift left 1 */                \
  /* q1: a0b1, a1b1, a2b1, a3b1 */                                        \
  "vext.8 q3, q13, q13, #8\n"  /* circular shift left 2 */                \
  /* q3: a0b2, a1b2, a2b2, a3b2 */                                        \
  "vext.8 q5, q15, q15, #4\n"  /* circular shift left 3 */                \
  /* q5: a0b3, a1b3, a2b3, a3b3 */                                        \
  /* trans 1 */                                                           \
  "vtrn.32    q8, q0\n"  /* get q8: a0b0, a0b1, a2b0, a2b1; */            \
  /* q0: a1b0, a1b1, a3b0, a3b1 */                                        \
  "vtrn.32    q2, q4\n"  /* get q2: a0b2, a0b3, a2b2, a2b3; */            \
  /* q4: a1b2, a1b3, a3b2, a3b3 */                                        \
  "vtrn.32    q9, q1\n"  /* get q9: a0b0, a0b1, a2b0, a2b1; */            \
  /* q1: a1b0, a1b1, a3b0, a3b1 */                                        \
  "vtrn.32    q3, q5\n"  /* get q3: a0b2, a0b3, a2b2, a2b3; */            \
  /* q5: a1b2, a1b3, a3b2, a3b3 */                                        \
  /* trans 2 */                                                           \
  "vswp   d17,    d4\n"  /* get q8: a0b0, a0b1, a0b2, a0b3; */            \
  /* q2: a2b0, a2b1, a2b2, a2b3 */                                        \
  "vswp   d1, d8\n"      /* get q0: a1b0, a1b1, a1b2, a1b3; */            \
  /* q4: a3b0, a3b1, a3b2, a3b3 */                                        \
  "vswp   d19,    d6\n"  /* get q9: a0b0, a0b1, a0b2, a0b3; */            \
  /* q3: a2b0, a2b1, a2b2, a2b3 */                                        \
  "vswp   d3, d10\n"     /* get q1: a1b0, a1b1, a1b2, a1b3; */            \
  /* q5: a3b0, a3b1, a3b2, a3b3 */

#define GEMM_INT8_TRANS_INT32_TO_FP32   \
  /* write output */              \
  "vld1.32 {d12-d13}, [%[scale]]\n" /* load scale */            \
  "vld1.32 {d14-d15}, [%[bias]]\n"  /* load bias */             \
  "vcvt.f32.s32   q10, q8\n"        /* r00, cvt int32 to fp32*/ \
  "vcvt.f32.s32   q11, q9\n"        /* r01, cvt int32 to fp32*/ \
  "vcvt.f32.s32   q12, q0\n"        /* r10, cvt int32 to fp32*/ \
  "vcvt.f32.s32   q13, q1\n"        /* r11, cvt int32 to fp32*/ \
  "vdup.32    q8, d14[0]\n"   \
  "vdup.32    q9, d14[0]\n"   \
  "vdup.32    q0, d14[1]\n"   \
  "vdup.32    q1, d14[1]\n"   \
  "vmla.f32 q8, q10, d12[0]\n"      /*  r00, mul scale */ \
  "vmla.f32 q9, q11, d12[0]\n"      /*  r01, mul scale */ \
  "vmla.f32 q0, q12, d12[1]\n"      /*  r10, mul scale */ \
  "vmla.f32 q1, q13, d12[1]\n"      /*  r11, mul scale */ \
  "vcvt.f32.s32   q10, q2\n"        /* r20, cvt int32 to fp32*/ \
  "vcvt.f32.s32   q11, q3\n"        /* r21, cvt int32 to fp32*/ \
  "vcvt.f32.s32   q12, q4\n"        /* r30, cvt int32 to fp32*/ \
  "vcvt.f32.s32   q13, q5\n"        /* r31, cvt int32 to fp32*/ \
  "vdup.32    q2, d15[0]\n"   \
  "vdup.32    q3, d15[0]\n"   \
  "vdup.32    q4, d15[1]\n"   \
  "vdup.32    q5, d15[1]\n"   \
  "vmla.f32 q2, q10, d13[0]\n"      /* r20, mul scale */  \
  "vmla.f32 q3, q11, d13[0]\n"      /* r21, mul scale */  \
  "vmla.f32 q4, q12, d13[1]\n"      /* r30, mul scale */  \
  "vmla.f32 q5, q13, d13[1]\n"      /* r31, mul scale */


#define GEMM_INT8_RELU  \
  /* do relu */       \
  "cmp    %[is_relu], #0\n"   /* skip relu */ \
  "beq    9f\n"               /* skip relu */ \
  "cmp        %[is_relu], #1\n"  /* check if has relu6 */  \
  "vmov.i32   q15, #0\n"      /* for relu */  \
  "bne    10f\n"               /* skip relu */ \
  "vmax.f32   q8, q8, q15\n"  /* relu */      \
  "vmax.f32   q9, q9, q15\n"  /* relu */      \
  "vmax.f32  q0,q0, q15\n"    /* relu */      \
  "vmax.f32  q1,q1, q15\n"    /* relu */      \
  "vmax.f32  q2,q2, q15\n"    /* relu */      \
  "vmax.f32  q3,q3, q15\n"    /* relu */      \
  "vmax.f32  q4,q4, q15\n"    /* relu */      \
  "vmax.f32  q5,q5, q15\n"    /* relu */      \
  "b  9f\n"

#define GEMM_INT8_RELU6  \
  /* do relu6 */       \
  "10: \n"             \
  "cmp    %[is_relu], #2\n"   /*heck if has relu6*/  \
  "bne    11f\n"               /* skip relu */ \
  "vmax.f32   q8, q8, q15\n"  /* relu */      \
  "vmax.f32   q9, q9, q15\n"  /* relu */      \
  "vmax.f32  q0,q0, q15\n"    /* relu */      \
  "vmax.f32  q1,q1, q15\n"    /* relu */      \
  "vld1.f32   {d28-d29}, [%[alpha]] @ load relu6 alpha\n" \
  "vmax.f32  q2,q2, q15\n"    /* relu */      \
  "vmax.f32  q3,q3, q15\n"    /* relu */      \
  "vmax.f32  q4,q4, q15\n"    /* relu */      \
  "vmax.f32  q5,q5, q15\n"    /* relu */      \
  "vmin.f32   q8, q8, q14\n"  /* relu6 */     \
  "vmin.f32   q9, q9, q14\n"  /* relu6 */     \
  "vmin.f32  q0,q0, q14\n"    /* relu6 */     \
  "vmin.f32  q1,q1, q14\n"    /* relu6 */     \
  "vmin.f32  q2,q2, q14\n"    /* relu6 */     \
  "vmin.f32  q3,q3, q14\n"    /* relu6 */     \
  "vmin.f32  q4,q4, q14\n"    /* relu6 */     \
  "vmin.f32  q5,q5, q14\n"    /* relu6 */     \
  "b  9f\n"

#define GEMM_INT8_LEAKY_RELU  \
  /* do leakyrelu */       \
  "11: \n"             \
  "cmp    %[is_relu], #3\n"   /*heck if has leakyrelu*/  \
  "bne    12f\n"               /* skip leakyrelu */ \
  "vld1.f32   {d28-d29}, [%[alpha]]       @ load relu6 alpha\n" \
  "vcge.f32   q6, q8, q15                @ vcgeq_u32 \n"    \
  "vmul.f32   q7, q8, q14                @ vmulq_f32 \n"    \
  "vcge.f32   q10, q9, q15                @ vcgeq_u32 \n"   \
  "vmul.f32   q11, q9, q14                @ vmulq_f32 \n"   \
  "vcge.f32   q12, q0, q15                @ vcgeq_u32 \n"   \
  "vmul.f32   q13, q0, q14                @ vmulq_f32 \n"   \
  "vbif       q8, q7, q6                @ choose    \n"     \
  "vbif       q9, q11, q10                @ choose    \n"   \
  "vbif       q0, q13, q12                @ choose    \n"   \
  "vcge.f32   q6, q1, q15                @ vcgeq_u32 \n"    \
  "vmul.f32   q7, q1, q14                @ vmulq_f32 \n"    \
  "vcge.f32   q10, q2, q15                @ vcgeq_u32 \n"   \
  "vmul.f32   q11, q2, q14                @ vmulq_f32 \n"   \
  "vcge.f32   q12, q3, q15                @ vcgeq_u32 \n"   \
  "vmul.f32   q13, q3, q14                @ vmulq_f32 \n"   \
  "vbif       q1, q7, q6                @ choose    \n"     \
  "vbif       q2, q11, q10                @ choose    \n"   \
  "vbif       q3, q13, q12                @ choose    \n"   \
  "vcge.f32   q6, q4, q15                @ vcgeq_u32 \n"    \
  "vmul.f32   q7, q4, q14                @ vmulq_f32 \n"    \
  "vcge.f32   q10, q5, q15                @ vcgeq_u32 \n"   \
  "vmul.f32   q11, q5, q14                @ vmulq_f32 \n"   \
  "vbif       q4, q7, q6                @ choose    \n"     \
  "vbif       q5, q11, q10                @ choose    \n"   \
  "b  9f\n"
#define GEMM_INT8_HARD_SWISH  \
  /* do hard_swish */       \
  "12: \n"             \
  "vldr       d24,  [%[alpha], #32]      @ load offset\n"   \
  "vldr       d25,  [%[alpha], #40]      @ load offset\n"   \
  "vld1.f32   {d28-d29}, [%[alpha]]       @ load relu6 alpha\n" \
  "vldr       d26,  [%[alpha], #48]      @ load threshold\n"\
  "vldr       d27,  [%[alpha], #56]      @ load threshold\n"\
  "vadd.f32   q6,  q8, q12                           \n"    \
  "vmul.f32   q8,  q8, q14                           \n"    \
  "vadd.f32   q7,  q9, q12                           \n"    \
  "vmul.f32   q9,  q9, q14                           \n"    \
  "vadd.f32   q10, q0, q12                           \n"    \
  "vmul.f32   q0,  q0, q14                           \n"    \
  "vadd.f32   q11, q1, q12                           \n"    \
  "vmul.f32   q1,  q1, q14                           \n"    \
  "vmax.f32   q6,  q6, q15                           \n"    \
  "vmax.f32   q7,  q7, q15                           \n"    \
  "vmax.f32   q10, q10, q15                          \n"    \
  "vmax.f32   q11, q11, q15                          \n"    \
  "vmin.f32   q6,  q6, q13                           \n"    \
  "vmin.f32   q7,  q7, q13                           \n"    \
  "vmin.f32   q10, q10, q13                          \n"    \
  "vmin.f32   q11, q11, q13                          \n"    \
  "vmul.f32   q8,  q8, q6                            \n"    \
  "vmul.f32   q9,  q9, q7                            \n"    \
  "vmul.f32   q0,  q0, q10                           \n"    \
  "vmul.f32   q1,  q1, q11                           \n"    \
  "vadd.f32   q6,  q2, q12                           \n"    \
  "vmul.f32   q2,  q2, q14                           \n"    \
  "vadd.f32   q7,  q3, q12                           \n"    \
  "vmul.f32   q3,  q3, q14                           \n"    \
  "vadd.f32   q10, q4, q12                           \n"    \
  "vmul.f32   q4,  q4, q14                           \n"    \
  "vadd.f32   q11, q5, q12                           \n"    \
  "vmul.f32   q5,  q5, q14                           \n"    \
  "vmax.f32   q6,  q6, q15                           \n"    \
  "vmax.f32   q7,  q7, q15                           \n"    \
  "vmax.f32   q10, q10, q15                          \n"    \
  "vmax.f32   q11, q11, q15                          \n"    \
  "vmin.f32   q6,  q6, q13                           \n"    \
  "vmin.f32   q7,  q7, q13                           \n"    \
  "vmin.f32   q10, q10, q13                          \n"    \
  "vmin.f32   q11, q11, q13                          \n"    \
  "vmul.f32   q2,  q2, q6                            \n"    \
  "vmul.f32   q3,  q3, q7                            \n"    \
  "vmul.f32   q4,  q4, q10                           \n"    \
  "vmul.f32   q5,  q5, q11                           \n"    \
  "9:  \n"

#define GEMM_INT8_FP32_OUT          \
  GEMM_INT8_TRANS_INT32_TO_FP32   \
  GEMM_INT8_RELU                  \
  GEMM_INT8_RELU6                 \
  GEMM_INT8_LEAKY_RELU            \
  GEMM_INT8_HARD_SWISH            \
  "vst1.32    {d16-d19},  [%[c_ptr0]]!\n" /* write r0, float32x4 x2 */ \
  "vst1.32    {d0-d3},    [%[c_ptr1]]!\n" /* write r1, float32x4 x2 */ \
  "vst1.32    {d4-d7},    [%[c_ptr2]]!\n" /* write r2, float32x4 x2 */ \
  "vst1.32    {d8-d11},   [%[c_ptr3]]!\n" /* write r3, float32x4 x2 */


#define GEMM_INT8_INT8_OUT      \
  GEMM_INT8_TRANS_INT32_TO_FP32   \
  GEMM_INT8_RELU                  \
  GEMM_INT8_RELU6                 \
  GEMM_INT8_LEAKY_RELU            \
  GEMM_INT8_HARD_SWISH            \
  "vmov.f32  q7, #-0.5\n"    /* neg offset */          \
  "vmov.f32  q10, #0.5\n"    /* pos offset */          \
  "vmov.f32  q11, #0.5\n"    /* pos offset */          \
  "vmov.f32  q12, #0.5\n"    /* pos offset */          \
  "vmov.f32  q13, #0.5\n"    /* pos offset */          \
  "vcgt.f32  q14, q8, #0\n"  /* get pos mask */        \
  "vcgt.f32  q15, q9, #0\n"  /* get pos mask */        \
  "vbif.f32  q10, q7, q14\n" /* get right offset */    \
  "vbif.f32  q11, q7, q15\n" /* get right offset */    \
  "vcgt.f32  q14, q0, #0\n"  /* get pos mask */        \
  "vcgt.f32  q15, q1, #0\n"  /* get pos mask */        \
  "vbif.f32  q12, q7, q14\n" /* get right offset */    \
  "vbif.f32  q13, q7, q15\n" /* get right offset */    \
  "vadd.f32 q8, q10, q8\n"   /* r00, add offset */     \
  "vadd.f32 q9, q11, q9\n"   /* r01, add offset */     \
  "vadd.f32 q0, q12, q0\n"   /* r10, add offset */     \
  "vadd.f32 q1, q13, q1\n"   /* r11, add offset */     \
  "vmov.f32  q10, #0.5\n"    /* pos offset */          \
  "vmov.f32  q11, #0.5\n"    /* pos offset */          \
  "vmov.f32  q12, #0.5\n"    /* pos offset */          \
  "vmov.f32  q13, #0.5\n"    /* pos offset */          \
  "vcgt.f32  q14, q2, #0\n"  /* get pos mask */        \
  "vcgt.f32  q15, q3, #0\n"  /* get pos mask */        \
  "vbif.f32  q10, q7, q14\n" /* get right offset */    \
  "vbif.f32  q11, q7, q15\n" /* get right offset */    \
  "vcgt.f32  q14, q4, #0\n"  /* get pos mask */        \
  "vcgt.f32  q15, q5, #0\n"  /* get pos mask */        \
  "vbif.f32  q12, q7, q14\n" /* get right offset */    \
  "vbif.f32  q13, q7, q15\n" /* get right offset */    \
  "add %[alpha], #16 \n"                               \
  "vadd.f32 q2, q10, q2\n"   /* r20, add offset */     \
  "vadd.f32 q3, q11, q3\n"   /* r21, add offset */     \
  "vadd.f32 q4, q12, q4\n"   /* r30, add offset */     \
  "vadd.f32 q5, q13, q5\n"   /* r31, add offset */     \
  "vld1.f32 {d12-d13}, [%[alpha]] \n"                  \
  "sub %[alpha], #16 \n"                               \
  "vcge.f32 q7, q8, q6\n"   /* @ q8 >= -127 \n */      \
  "vcge.f32 q10, q9, q6\n"   /* @ q9 >= -127 \n */     \
  "vcge.f32 q11, q0, q6\n"   /* @ q0 >= -127 \n */     \
  "vcge.f32 q12, q1, q6\n"   /* @ q1 >= -127 \n */     \
  "vcge.f32 q13, q2, q6\n"   /* @ q2 >= -127 \n */     \
  "vcge.f32 q14, q3, q6\n"   /* @ q3 >= -127 \n */     \
  "vcge.f32 q15, q4, q6\n"   /* @ q4 >= -127 \n */     \
  /* choose data */                                    \
  "vbif q8, q6, q7\n"       /* @ choose */            \
  "vcge.f32 q7, q5, q6\n"   /* @ q5 >= -127 \n */     \
  "vbif q9, q6, q10\n"       /* @ choose */             \
  "vbif q0, q6, q11\n"       /* @ choose */           \
  "vbif q1, q6, q12\n"       /* @ choose */           \
  "vbif q2, q6, q13\n"       /* @ choose */           \
  "vbif q3, q6, q14\n"       /* @ choose */           \
  "vbif q4, q6, q15\n"       /* @ choose */           \
  "vbif q5, q6, q7\n"       /* @ choose */           \
  "vcvt.s32.f32   q6, q8\n"  /* r00, fp32->int32 */    \
  "vcvt.s32.f32   q7, q9\n"  /* r01, fp32->int32 */    \
  "vcvt.s32.f32   q10, q0\n" /* r10, fp32->int32 */    \
  "vcvt.s32.f32   q11, q1\n" /* r11, fp32->int32 */    \
  "vcvt.s32.f32   q12, q2\n" /* r20, fp32->int32 */    \
  "vcvt.s32.f32   q13, q3\n" /* r21, fp32->int32 */    \
  "vcvt.s32.f32   q14, q4\n" /* r30, fp32->int32 */    \
  "vcvt.s32.f32   q15, q5\n" /* r31, fp32->int32 */    \
  "vqmovn.s32 d0, q6\n"      /* r00, int32 -> int16 */ \
  "vqmovn.s32 d1, q7\n"      /* r01, int32 -> int16 */ \
  "vqmovn.s32 d2, q10\n"     /* r10, int32 -> int16 */ \
  "vqmovn.s32 d3, q11\n"     /* r11, int32 -> int16 */ \
  "vqmovn.s32 d4, q12\n"     /* r00, int32 -> int16 */ \
  "vqmovn.s32 d5, q13\n"     /* r01, int32 -> int16 */ \
  "vqmovn.s32 d6, q14\n"     /* r10, int32 -> int16 */ \
  "vqmovn.s32 d7, q15\n"     /* r11, int32 -> int16 */ \
  "vqmovn.s16 d8, q0\n"      /* 0, int16 -> int8 */    \
  "vqmovn.s16 d9, q1\n"      /* 1, int16 -> int8 */    \
  "vqmovn.s16 d10, q2\n"     /* 2, int16 -> int8 */    \
  "vqmovn.s16 d11, q3\n"     /* 3, int16 -> int8 */    \
  "vst1.32    {d8}, [%[c_ptr0]]!\n"  /* write r0*/     \
  "vst1.32    {d9}, [%[c_ptr1]]!\n"  /* write r1*/     \
  "vst1.32    {d10}, [%[c_ptr2]]!\n" /* write r2*/     \
  "vst1.32    {d11}, [%[c_ptr3]]!\n" /* write r3*/

#define GEMM_INT8_INT32_OUT                            \
  "vst1.32    {d16-d19},  [%[c_ptr0]]!\n" /* write r0, float32x4 x2 */ \
  "vst1.32    {d0-d3},    [%[c_ptr1]]!\n" /* write r1, float32x4 x2 */ \
  "vst1.32    {d4-d7},    [%[c_ptr2]]!\n" /* write r2, float32x4 x2 */ \
  "vst1.32    {d8-d11},   [%[c_ptr3]]!\n" /* write r3, float32x4 x2 */

// clang-format on

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,  // NOLINT
                             const float* bias,
                             float32_t*& c_ptr0,  // NOLINT
                             float32_t*& c_ptr1,  // NOLINT
                             float32_t*& c_ptr2,  // NOLINT
                             float32_t*& c_ptr3,  // NOLINT
                             const float32_t* scale,
                             const float32_t* alpha,
                             int is_relu,
                             int k,
                             int rem) {
  float new_ptr[16] = {alpha[0],
                       alpha[1],
                       alpha[2],
                       alpha[3],
                       -127.0,
                       -127.0,
                       -127.0,
                       -127.0,
                       alpha[4],
                       alpha[5],
                       alpha[6],
                       alpha[7],
                       alpha[8],
                       alpha[9],
                       alpha[10],
                       alpha[11]};

  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_FP32_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [k] "+r"(k)
               : [is_relu] "r"(is_relu),
                 [bias] "r"(bias),
                 [alpha] "r"(new_ptr),
                 [rem] "r"(rem),
                 [scale] "r"(scale)
               : "q0",
                 "q1",
                 "q2",
                 "q3",
                 "q4",
                 "q5",
                 "q6",
                 "q7",
                 "q8",
                 "q9",
                 "q10",
                 "q11",
                 "q12",
                 "q13",
                 "q14",
                 "q15",
                 "cc",
                 "memory");
}

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,  // NOLINT
                             const float* bias,
                             int8_t*& c_ptr0,  // NOLINT
                             int8_t*& c_ptr1,  // NOLINT
                             int8_t*& c_ptr2,  // NOLINT
                             int8_t*& c_ptr3,  // NOLINT
                             const float32_t* scale,
                             const float32_t* alpha,
                             int is_relu,
                             int k,
                             int rem) {
  float new_ptr[16] = {alpha[0],
                       alpha[1],
                       alpha[2],
                       alpha[3],
                       -127.0,
                       -127.0,
                       -127.0,
                       -127.0,
                       alpha[4],
                       alpha[5],
                       alpha[6],
                       alpha[7],
                       alpha[8],
                       alpha[9],
                       alpha[10],
                       alpha[11]};
  // clang-format off
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_INT8_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [k] "+r"(k)
               : [is_relu] "r"(is_relu),
                 [alpha] "r"(new_ptr),
                 [bias] "r"(bias),
                 [rem] "r"(rem),
                 [scale] "r"(scale)
               : "q0",
                 "q1",
                 "q2",
                 "q3",
                 "q4",
                 "q5",
                 "q6",
                 "q7",
                 "q8",
                 "q9",
                 "q10",
                 "q11",
                 "q12",
                 "q13",
                 "q14",
                 "q15",
                 "cc",
                 "memory");
  // clang-format on
}

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,  // NOLINT
                             const float* bias,
                             int32_t*& c_ptr0,  // NOLINT
                             int32_t*& c_ptr1,  // NOLINT
                             int32_t*& c_ptr2,  // NOLINT
                             int32_t*& c_ptr3,  // NOLINT
                             const float32_t* scale,
                             const float32_t* alpha,
                             int is_relu,
                             int k,
                             int rem) {
  float new_ptr[16] = {alpha[0],
                       alpha[1],
                       alpha[2],
                       alpha[3],
                       -127.0,
                       -127.0,
                       -127.0,
                       -127.0,
                       alpha[4],
                       alpha[5],
                       alpha[6],
                       alpha[7],
                       alpha[8],
                       alpha[9],
                       alpha[10],
                       alpha[11]};
  // clang-format off
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_INT32_OUT
               : [a_ptr] "+r"(a_ptr),
                 [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0),
                 [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2),
                 [c_ptr3] "+r"(c_ptr3),
                 [k] "+r"(k)
               : [is_relu] "r"(is_relu),
                 [alpha] "r"(new_ptr),
                 [bias] "r"(bias),
                 [rem] "r"(rem),
                 [scale] "r"(scale)
               : "q0",
                 "q1",
                 "q2",
                 "q3",
                 "q4",
                 "q5",
                 "q6",
                 "q7",
                 "q8",
                 "q9",
                 "q10",
                 "q11",
                 "q12",
                 "q13",
                 "q14",
                 "q15",
                 "cc",
                 "memory");
  // clang-format on
}
#endif  // __aarch64__ // NOLINT

// gemm wrapper
template <typename Dtype>
void gemm_prepack_oth_int8(const int8_t* A_packed,
                           const int8_t* B,
                           const float* bias,
                           Dtype* C,
                           int M,
                           int N,
                           int K,
                           bool is_bias,
                           int flag_act,
                           bool is_transB,
                           const float* scale,
                           const float* alpha,
                           ARMContext* ctx) {
  const int KUP = ROUNDUP(K, KBLOCK_INT8);
  size_t llc_size = ctx->llc_size() / 4;
  auto workspace = ctx->workspace_data<int8_t>();
  int threads = ctx->threads();
  int x_block = llc_size / (sizeof(int8_t) * (KUP + MBLOCK_INT8_OTH));
  x_block /= NBLOCK_INT8_OTH;
  x_block *= NBLOCK_INT8_OTH;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK_INT8_OTH - 1) / NBLOCK_INT8_OTH;
  x_block *= NBLOCK_INT8_OTH;
  int k = K / KBLOCK_INT8;
  int k_rem = K & (KBLOCK_INT8 - 1);
  if (k_rem > KBLOCK_INT8 / 2) {
    k_rem = 0;
    k += 1;
  }
  int n_rem = N & (NBLOCK_INT8_OTH - 1);

  auto* b_tmp = static_cast<int8_t*>(workspace);

  auto* zerobuf =
      static_cast<int8_t*>(malloc(x_block * (sizeof(int8_t) + sizeof(Dtype))));
  memset(zerobuf, 0, x_block * sizeof(int8_t));
  auto* trash_ptr =
      reinterpret_cast<Dtype*>(zerobuf + x_block * sizeof(int8_t));

  //! apanel is pre_compute outside gemm

  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    bool flag_rem = false;
    if (xmax >= N) {
      xmax = N;
      flag_rem = n_rem > 0;
    }
    int bblocks = (xmax - x0 + NBLOCK_INT8_OTH - 1) / NBLOCK_INT8_OTH;
    //! load bpanel
    int8_t* b_pannel = b_tmp;
    if (is_transB) {
      packb_trans_int8(b_pannel, B, K, 0, K, x0, xmax, zerobuf);
    } else {
      packb_int8(b_pannel, B, N, 0, K, x0, xmax, zerobuf);
    }

    LITE_PARALLEL_COMMON_BEGIN(y, tid, M, 0, MBLOCK_INT8_OTH) {
      Dtype out0[NBLOCK_INT8_OTH] = {0};
      Dtype out1[NBLOCK_INT8_OTH] = {0};
      Dtype out2[NBLOCK_INT8_OTH] = {0};
      Dtype out3[NBLOCK_INT8_OTH] = {0};
      Dtype* c_ptr0 = C + y * N + x0;
      Dtype* c_ptr1 = c_ptr0 + N;
      Dtype* c_ptr2 = c_ptr1 + N;
      Dtype* c_ptr3 = c_ptr2 + N;
      Dtype* tmp0 = nullptr;
      Dtype* tmp1 = nullptr;
      Dtype* tmp2 = nullptr;
      Dtype* tmp3 = nullptr;
      float32_t scale_local[4] = {0, 0, 0, 0};
      float32_t bias_local[4] = {0, 0, 0, 0};
      if (is_bias) {
        if (y + 4 <= M) {
          bias_local[0] = bias[y];
          bias_local[1] = bias[y + 1];
          bias_local[2] = bias[y + 2];
          bias_local[3] = bias[y + 3];
        } else {
          switch (M - y) {
            case 3:
              bias_local[2] = bias[y + 2];
            case 2:
              bias_local[1] = bias[y + 1];
            case 1:
              bias_local[0] = bias[y + 0];
            default:
              break;
          }
        }
      }
      if (scale) {
        if (y + 4 <= M) {
          scale_local[0] = scale[y];
          scale_local[1] = scale[y + 1];
          scale_local[2] = scale[y + 2];
          scale_local[3] = scale[y + 3];
        } else {
          switch (M - y) {
            case 3:
              scale_local[2] = scale[y + 2];
            case 2:
              scale_local[1] = scale[y + 1];
            case 1:
              scale_local[0] = scale[y + 0];
            default:
              break;
          }
        }
      }
      if (y + MBLOCK_INT8_OTH > M) {
        switch (y + MBLOCK_INT8_OTH - M) {
          case 3:
            c_ptr1 = trash_ptr;
          case 2:
            c_ptr2 = trash_ptr;
          case 1:
            c_ptr3 = trash_ptr;
          default:
            break;
        }
      }
      const int8_t* a_ptr_l = A_packed + y * KUP;
      const int8_t* b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if (flag_rem && (xb == bblocks - 1)) {
          tmp0 = c_ptr0;
          tmp1 = c_ptr1;
          tmp2 = c_ptr2;
          tmp3 = c_ptr3;
          c_ptr0 = out0;
          c_ptr1 = out1;
          c_ptr2 = out2;
          c_ptr3 = out3;
        }
        gemm_int8_kernel<Dtype>(a_ptr_l,
                                b_ptr,
                                bias_local,
                                c_ptr0,
                                c_ptr1,
                                c_ptr2,
                                c_ptr3,
                                scale_local,
                                alpha,
                                flag_act,
                                k,
                                k_rem);
        if (flag_rem && (xb == bblocks - 1)) {
          for (int i = 0; i < n_rem; ++i) {
            *(tmp0++) = out0[i];
            *(tmp1++) = out1[i];
            *(tmp2++) = out2[i];
            *(tmp3++) = out3[i];
          }
        }
      }
    }
    LITE_PARALLEL_COMMON_END();
  }
  free(zerobuf);
}

/***********************************************************************/
// prepack A according to gemm kernel
// A block size: (<4x2>x1) x2, with unroll=2 can be described as below:
// origin A data:
// A_origin(no trans, m x k):
//      r0: ==>   a0, b0, c0, d0, e0, f0, g0, h0
//      r1: ==>   a1, b1, c1, d1, e1, f1, g1, h1
//      r2: ==>   a2, b2, c2, d2, e2, f2, g2, h2
//      r3: ==>   a3, b3, c3, d3, e3, f3, g3, h3
// packed A
//      a0,b0, a1,b1, a2,b2, a3,b3;
//      c0,d0, c1,d1, c2,d2, c3,d3;
//      e0,f0, e1,f1, e2,f2, e3,f3;
//      g0,h0, g1,h1, g2,h2, g3,h3;
/***********************************************************************/
void prepackA_m4k2x2_int8(int8_t* out,
                          const int8_t* in,
                          const int ldin,
                          const int m0,
                          const int mmax,
                          const int k0,
                          const int kmax) {
  int y_len = mmax - m0;
  int x_len = kmax - k0;
  int x_len_roundup = ROUNDUP(x_len, KBLOCK_INT8);
  auto zerobuff = static_cast<int8_t*>(malloc(x_len_roundup * sizeof(char)));
  memset(zerobuff, 0, sizeof(char) * x_len_roundup);

  const int8_t* inptr = in + m0 * ldin + k0;
  uint8_t remain = static_cast<uint8_t>(x_len & (KBLOCK_INT8 - 1));

  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, 0, MBLOCK_INT8_OTH) {
    const int8_t* ptr0 = inptr + y * ldin;
    const int8_t* ptr1 = ptr0 + ldin;
    const int8_t* ptr2 = ptr1 + ldin;
    const int8_t* ptr3 = ptr2 + ldin;
    //! cope with row index exceed real size, set to zero buffer
    if ((y + MBLOCK_INT8_OTH) > y_len) {
      switch ((y + MBLOCK_INT8_OTH) - y_len) {
        case 3:
          ptr1 = zerobuff;
        case 2:
          ptr2 = zerobuff;
        case 1:
          ptr3 = zerobuff;
        default:
          break;
      }
    }
    int8_t* ptr_out = out + y * x_len_roundup;
    int i = 0;
    for (; i < x_len + 1 - 2 * KBLOCK_INT8; i += 2 * KBLOCK_INT8) {
// clang-format off
#ifdef __aarch64__
      asm volatile(
          "ld1    {v0.8b}, [%[ptr0]], #8\n" /* load r0, 8 int8 */
          "ld1    {v1.8b}, [%[ptr1]], #8\n" /* load r1, 8 int8 */
          "ld1    {v2.8b}, [%[ptr2]], #8\n" /* load r2, 8 int8 */
          "ld1    {v3.8b}, [%[ptr3]], #8\n" /* load r3, 8 int8 */
          "trn1   v4.4h, v0.4h, v1.4h\n"    /* get a0,b0, a2,b2 */
          "trn2   v5.4h, v0.4h, v1.4h\n"    /* get a1,b1, a3,b3 */
          "trn1   v6.4h, v2.4h, v3.4h\n"    /* get c0,d0, c2,d2 */
          "trn2   v7.4h, v2.4h, v3.4h\n"    /* get c1,d1, c3,d3 */
          "trn1   v0.2s, v4.2s, v6.2s\n"    /* get a0,b0, c0,d0 */
          "trn2   v2.2s, v4.2s, v6.2s\n"    /* get a2,b2, c2,d2 */
          "trn1   v1.2s, v5.2s, v7.2s\n"    /* get a1,b1, c1,d1 */
          "trn2   v3.2s, v5.2s, v7.2s\n"    /* get a3,b3, c3,d3 */
          "st1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%[ptr_out]], #32\n" /* write
                                                                        out*/
          : [ptr_out] "+r"(ptr_out), [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2), [ptr3] "+r"(ptr3)
          :
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
#else   // armv7
      asm volatile(
          "vld1.8 {d0}, [%[ptr0]]!\n" /* load r0, 8 int8,
                                         a0,b0,c0,d0,e0,f0,g0,h0 */
          "vld1.8 {d1}, [%[ptr1]]!\n" /* load r1, 8 int8,
                                         a1,b1,c1,d1,e1,f1,g1,h1 */
          "vld1.8 {d2}, [%[ptr2]]!\n" /* load r2, 8 int8,
                                         a2,b2,c2,d2,e2,f2,g2,h2 */
          "vld1.8 {d3}, [%[ptr3]]!\n" /* load r3, 8 int8,
                                         a3,b3,c3,d3,e3,f3,g3,h3 */
          "vtrn.16    d0, d1\n" /* trans, d0: a0,b0,a1,b1, e0,f0,e1,f1; d1:
                                   c0,d0,c1,d1, g0,h0,g1,h1 */
          "vtrn.16    d2, d3\n" /* trans, d2: a2,b2,a3,b3, e2,f2,e3,f3; d3:
                                   c2,d2,c3,d3, g2,h2,g3,h3 */
          "vtrn.32    d0, d2\n" /* trans, d0: a0,b0,a1,b1, a2,b2,a3,b3; d2:
                                   e0,f0,e1,f1, e2,f2,e3,f3 */
          "vtrn.32    d1, d3\n" /* trans, d1: c0,d0,c1,d1, e2,f2,e3,f3; d3:
                                   g0,h0,g1,h1, g2,h2,g3,h3 */
          "vst1.32 {d0-d3}, [%[outptr]]!\n" /* write to output ptr */
          : [outptr] "+r"(ptr_out), [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2), [ptr3] "+r"(ptr3)
          :
          : "q0", "q1", "cc", "memory");
#endif  // __aarch64__
      // clang-format on
    }
    if (i + KBLOCK_INT8 <= x_len) {
      ptr_out[0] = ptr0[0];
      ptr_out[1] = ptr0[1];
      ptr_out[2] = ptr1[0];
      ptr_out[3] = ptr1[1];
      ptr_out[4] = ptr2[0];
      ptr_out[5] = ptr2[1];
      ptr_out[6] = ptr3[0];
      ptr_out[7] = ptr3[1];
      // unroll
      ptr_out[8] = ptr0[2];
      ptr_out[9] = ptr0[3];
      ptr_out[10] = ptr1[2];
      ptr_out[11] = ptr1[3];
      ptr_out[12] = ptr2[2];
      ptr_out[13] = ptr2[3];
      ptr_out[14] = ptr3[2];
      ptr_out[15] = ptr3[3];
      ptr_out += 16;
      ptr0 += 4;
      ptr1 += 4;
      ptr2 += 4;
      ptr3 += 4;
    }
    switch (remain) {
      case 0:
        break;
      case 1:
        ptr_out[0] = ptr0[0];
        ptr_out[1] = 0;
        ptr_out[2] = ptr1[0];
        ptr_out[3] = 0;
        ptr_out[4] = ptr2[0];
        ptr_out[5] = 0;
        ptr_out[6] = ptr3[0];
        ptr_out[7] = 0;
        // unroll
        ptr_out[8] = 0;
        ptr_out[9] = 0;
        ptr_out[10] = 0;
        ptr_out[11] = 0;
        ptr_out[12] = 0;
        ptr_out[13] = 0;
        ptr_out[14] = 0;
        ptr_out[15] = 0;
        ptr_out += 16;
        break;
      case 2:
        ptr_out[0] = ptr0[0];
        ptr_out[1] = ptr0[1];
        ptr_out[2] = ptr1[0];
        ptr_out[3] = ptr1[1];
        ptr_out[4] = ptr2[0];
        ptr_out[5] = ptr2[1];
        ptr_out[6] = ptr3[0];
        ptr_out[7] = ptr3[1];
        // unroll
        ptr_out[8] = 0;
        ptr_out[9] = 0;
        ptr_out[10] = 0;
        ptr_out[11] = 0;
        ptr_out[12] = 0;
        ptr_out[13] = 0;
        ptr_out[14] = 0;
        ptr_out[15] = 0;
        ptr_out += 16;
        break;
      case 3:
        ptr_out[0] = ptr0[0];
        ptr_out[1] = ptr0[1];
        ptr_out[2] = ptr1[0];
        ptr_out[3] = ptr1[1];
        ptr_out[4] = ptr2[0];
        ptr_out[5] = ptr2[1];
        ptr_out[6] = ptr3[0];
        ptr_out[7] = ptr3[1];
        // unroll
        ptr_out[8] = ptr0[2];
        ptr_out[9] = 0;
        ptr_out[10] = ptr1[2];
        ptr_out[11] = 0;
        ptr_out[12] = ptr2[2];
        ptr_out[13] = 0;
        ptr_out[14] = ptr3[2];
        ptr_out[15] = 0;
        ptr_out += 16;
        break;
      default:
        break;
    }
  }
  LITE_PARALLEL_COMMON_END();
  free(zerobuff);
}

/***************************************************************************/
// prepack A according to gemm kernel
// A block size: <4x2>x2, unroll x4, can be described as below:
// origin A data:
// A_origin(no trans, k x m):
//      r0: ==>   a0, a1, a2, a3 .... a12, a13, a14, a15
//      r1: ==>   b0, b1, b2, b3 .... b12, b13, b14, b15
//      r2: ==>   c0, c1, c2, c3 .... c12, c13, c14, c15
//      r3: ==>   d0, d1, d2, d3 .... d12, d13, d14, d15
// packed A:
//      a0,b0, a1,b1, a2,b2, a3,b3;
//      c0,d0, c1,d1, c2,d2, c3,d3;----block0
//      a4,b4, a5,b5, a6,b6, a7,b7;
//      c4,d4, c5,d5, c6,d6, c7,d7;----block1
//      a8,b8, a9,b9, a10,b10, a11,b11;
//      c8,d8, c9,d9, c10,d10, c11,d11;----block2
//      a12,b12, a13,b13, a14,b14, a15,b15;
//      c12,d12, c13,d13, c14,d14, c15,d15;----block3
/***************************************************************************/
void prepackA_m4k2x2_trans_int8(int8_t* out,
                                const int8_t* in,
                                const int ldin,
                                const int m0,
                                const int mmax,
                                const int k0,
                                const int kmax) {
  int xlen = mmax - m0;
  int ylen = kmax - k0;
  int ylen_roundup = ROUNDUP(ylen, KBLOCK_INT8);
  int xlen_roundup = ROUNDUP(xlen, MBLOCK_INT8_OTH);

  const int MUNROLL = 4;
  int mcnt = xlen / (MUNROLL * MBLOCK_INT8_OTH);
  int x_rem = xlen & (MUNROLL * MBLOCK_INT8_OTH - 1);
  int m_rem = (x_rem + MBLOCK_INT8_OTH - 1) / MBLOCK_INT8_OTH;

  const uint8_t mask_buffer[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int8x16_t vzero = vdupq_n_s8(0);
  uint8x16_t vmask = vcltq_u8(vld1q_u8(mask_buffer), vdupq_n_u8(x_rem));

  int stride_out = ylen_roundup * MBLOCK_INT8_OTH;

  int8_t* zerobuf = static_cast<int8_t*>(malloc(xlen_roundup));
  memset(zerobuf, 0, xlen_roundup);

  const int8_t* inr = in + ldin * k0 + m0;

  LITE_PARALLEL_COMMON_BEGIN(y, tid, ylen, 0, KBLOCK_INT8) {
    const int8_t* ptr0 = inr + y * ldin;
    const int8_t* ptr1 = ptr0 + ldin;
    const int8_t* ptr2 = ptr1 + ldin;
    const int8_t* ptr3 = ptr2 + ldin;
    int8_t* ptr_out = out + MBLOCK_INT8_OTH * y;
    if (y + KBLOCK_INT8 > ylen) {
      switch (y + KBLOCK_INT8 - ylen) {
        case 3:
          ptr1 = zerobuf;
        case 2:
          ptr2 = zerobuf;
        case 1:
          ptr3 = zerobuf;
        default:
          break;
      }
    }
    int k = mcnt;
    int rem = m_rem;
// clang-format off
#ifdef __aarch64__
    asm volatile(
        "ld1    {v0.16b},   [%[ptr0]],  #16\n" /* load r0 */
        "ld1    {v1.16b},   [%[ptr1]],  #16\n" /* load r1 */
        "ld1    {v2.16b},   [%[ptr2]],  #16\n" /* load r2 */
        "ld1    {v3.16b},   [%[ptr3]],  #16\n" /* load r3 */
        "cbz    %w[k], 1f\n"                   /* jump to remain */
        "0:\n"                                 /* main loop */
        /* trans 16b */
        "trn1   v4.16b, v0.16b, v1.16b\n" /* get a0,b0, a2,b2, a4,b4, a6,b6,
                                             a8,b8, a10,b10, a12,b12, a14,b14 */
        "trn2   v5.16b, v0.16b, v1.16b\n" /* get a1,b1, a3,b3, a5,b5, a7,b7,
                                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "trn1   v6.16b, v2.16b, v3.16b\n" /* get c0,d0, c2,d2, c4,d4, c6,d6,
                                             c8,d8, c10,d10, c12,d12, c14,d14 */
        "trn2   v7.16b, v2.16b, v3.16b\n" /* get c1,d1, c3,d3, c5,d5, c7,d7,
                                             c9,d9, c11,d11, c13,d13, c15,d15 */
        "ld1    {v0.16b},   [%[ptr0]],  #16\n" /* load r0 */
        "ld1    {v1.16b},   [%[ptr1]],  #16\n" /* load r1 */
        "subs   %w[k], %w[k], #1\n"            /* loop cnt -1 */
        /* trans 8h */
        "trn1   v8.8h, v4.8h, v5.8h\n" /* get a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                          a9,b9, a12,b12, a13,b13 */
        "trn2   v9.8h, v4.8h, v5.8h\n" /* get a2,b2, a3,b3, a6,b6, a7,b7,
                                          a10,b10, a11,b11, a14,b14, a15,b15 */
        "trn1   v10.8h, v6.8h, v7.8h\n" /* get c0,d0, c1,d1, c4,d4, c5,d5,
                                           c8,d8, c9,d9, c12,d12, c13,d13 */
        "trn2   v11.8h, v6.8h, v7.8h\n" /* get c2,d2, c3,d3, c6,d6, c7,d7,
                                           c10,d10, c11,d11, c14,d14, c15,d15 */
        /* trans 4s */
        "ld1    {v2.16b},   [%[ptr2]],  #16\n" /* load r2 */
        "trn1   v4.4s, v8.4s, v9.4s\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                          a9,b9, a10,b10, a11,b11 */
        "trn2   v5.4s, v8.4s, v9.4s\n" /* get a4,b4, a5,b5, a6,b6, a7,b7,
                                          a12,b12, a13,b13, a14,b14, a15,b15 */
        "trn1   v6.4s, v10.4s, v11.4s\n" /* get c0,d0, c1,d1, c2,d2, c3,d3,
                                            c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v7.4s, v10.4s, v11.4s\n" /* get c4,d4, c5,d5, c6,d6, c7,d7,
                                            c12,d12, c13,d13, c14,d14, c15,d15
                                            */
        /* trans 2d */
        "ld1    {v3.16b},   [%[ptr3]],  #16\n" /* load r3 */
        "trn1   v8.2d, v4.2d, v6.2d\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, c0,d0,
                                          c1,d1, c2,d2, c3,d3 */
        "trn1   v9.2d, v5.2d, v7.2d\n" /* get a4,b4, a5,b5, a6,b6, a7,b7, c4,d4,
                                          c5,d5, c6,d6, c7,d7 */
        "trn2   v10.2d, v4.2d, v6.2d\n" /* get a8,b8, a9,b9, a10,b10, a11,b11,
                                           c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v11.2d, v5.2d, v7.2d\n" /* get a12,b12, a13,b13, a14,b14,
                                           a15,b15, c12,d12, c13,d13, c14,d14,
                                           c15,d15 */
        "st1    {v8.16b}, [%[ptr_out]], %[stride]\n" /* write block0, address +
                                                        stride */
        "st1    {v9.16b}, [%[ptr_out]], %[stride]\n" /* write block1, address +
                                                        stride */
        "st1   {v10.16b}, [%[ptr_out]], %[stride]\n" /* write block2, address +
                                                        stride */
        "st1   {v11.16b}, [%[ptr_out]], %[stride]\n" /* write block3, address +
                                                        stride */
        "bgt    0b\n"                                /* jump to main loop */
        "1:\n"                                       /* process remain */
        "cbz    %w[rem], 2f\n"                       /* skip to remain */
        /* bit select */
        "bif    v0.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v1.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v2.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v3.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        /* trans 16b */
        "trn1   v4.16b, v0.16b, v1.16b\n" /* get a0,b0, a2,b2, a4,b4, a6,b6,
                                             a8,b8, a10,b10, a12,b12, a14,b14 */
        "trn2   v5.16b, v0.16b, v1.16b\n" /* get a1,b1, a3,b3, a5,b5, a7,b7,
                                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "trn1   v6.16b, v2.16b, v3.16b\n" /* get c0,d0, c2,d2, c4,d4, c6,d6,
                                             c8,d8, c10,d10, c12,d12, c14,d14 */
        "trn2   v7.16b, v2.16b, v3.16b\n" /* get c1,d1, c3,d3, c5,d5, c7,d7,
                                             c9,d9, c11,d11, c13,d13, c15,d15 */
        /* trans 8h */
        "trn1   v8.8h, v4.8h, v5.8h\n" /* get a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                          a9,b9, a12,b12, a13,b13 */
        "trn2   v9.8h, v4.8h, v5.8h\n" /* get a2,b2, a3,b3, a6,b6, a7,b7,
                                          a10,b10, a11,b11, a14,b14, a15,b15 */
        "trn1   v10.8h, v6.8h, v7.8h\n" /* get c0,d0, c1,d1, c4,d4, c5,d5,
                                           c8,d8, c9,d9, c12,d12, c13,d13 */
        "trn2   v11.8h, v6.8h, v7.8h\n" /* get c2,d2, c3,d3, c6,d6, c7,d7,
                                           c10,d10, c11,d11, c14,d14, c15,d15 */
        /* trans 4s */
        "trn1   v4.4s, v8.4s, v9.4s\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                          a9,b9, a10,b10, a11,b11 */
        "trn2   v5.4s, v8.4s, v9.4s\n" /* get a4,b4, a5,b5, a6,b6, a7,b7,
                                          a12,b12, a13,b13, a14,b14, a15,b15 */
        "trn1   v6.4s, v10.4s, v11.4s\n" /* get c0,d0, c1,d1, c2,d2, c3,d3,
                                            c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v7.4s, v10.4s, v11.4s\n" /* get c4,d4, c5,d5, c6,d6, c7,d7,
                                            c12,d12, c13,d13, c14,d14, c15,d15
                                            */
        /* trans 2d */
        "trn1   v8.2d, v4.2d, v6.2d\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, c0,d0,
                                          c1,d1, c2,d2, c3,d3 */
        "trn1   v9.2d, v5.2d, v7.2d\n" /* get a4,b4, a5,b5, a6,b6, a7,b7, c4,d4,
                                          c5,d5, c6,d6, c7,d7 */
        "trn2   v10.2d, v4.2d, v6.2d\n" /* get a8,b8, a9,b9, a10,b10, a11,b11,
                                           c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v11.2d, v5.2d, v7.2d\n" /* get a12,b12, a13,b13, a14,b14,
                                           a15,b15, c12,d12, c13,d13, c14,d14,
                                           c15,d15 */
        /* check remain size */
        "subs    %w[rem], %w[rem], #1\n"             /* check remain num */
        "st1    {v8.16b}, [%[ptr_out]], %[stride]\n" /* write 0 */
        "beq    2f\n"                                /* remain = 1 */
        "subs    %w[rem], %w[rem], #1\n"             /* check remain num */
        "st1    {v9.16b}, [%[ptr_out]], %[stride]\n" /* write 1 */
        "beq    2f\n"                                /* remain = 2 */
        "subs    %w[rem], %w[rem], #1\n"             /* check remain num */
        "st1   {v10.16b}, [%[ptr_out]], %[stride]\n" /* write 2 */
        "beq    2f\n"                                /* remain = 3 */
        "st1   {v11.16b}, [%[ptr_out]]\n"            /* write 3 */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [k] "+r"(k), [rem] "+r"(rem),
          [ptr_out] "+r"(ptr_out)
        : [mask] "w"(vmask), [vzero] "w"(vzero), [stride] "r"(stride_out)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
          "v11", "cc");
#else   // armv7
    asm volatile(
        "vld1.8 {d0-d1},    [%[ptr0]]!\n" /* load r0 */
        "vld1.8 {d2-d3},    [%[ptr1]]!\n" /* load r1 */
        "vld1.8 {d4-d5},    [%[ptr2]]!\n" /* load r2 */
        "vld1.8 {d6-d7},    [%[ptr3]]!\n" /* load r3 */
        "cmp    %[k], #0\n"               /* check main loop */
        "beq    1f\n"                     /* jump to remain */
        "0:\n"                            /* main loop */
        /* trans 16b */
        "vtrn.8 q0, q1\n" /* get q0: a0,b0, a2,b2, a4,b4, a6,b6, a8,b8, a10,b10,
                             a12,b12, a14,b14; q1: a1,b1, a3,b3, a5,b5, a7,b7,
                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "vtrn.8 q2, q3\n" /* get q2: c0,d0, c2,d2, c4,d4, c6,d6, c8,d8, c10,d10,
                             c12,d12, c14,d14; q3: c0,d0, c2,d2, c4,d4, c6,d6,
                             c8,d8, c10,d10, c12,d12, c14,d14 */
        "subs   %[k], %[k], #1\n" /* loop cnt -1 */
        /* trans 8h */
        "vtrn.16    q0, q1\n" /* get q0: a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                 a9,b9, a12,b12, a13,b13; q1: a2,b2, a3,b3,
                                 a6,b6, a7,b7, a10,b10, a11,b11, a14,b14,
                                 a15,b15 */
        "vtrn.16    q2, q3\n" /* get q2: c0,d0, c1,d1, c4,d4, c5,d5, c8,d8,
                                 c9,d9, c12,d12, c13,d13; q3: c2,d2, c3,d3,
                                 c6,d6, c7,d7, c10,d10, c11,d11, c14,d14,
                                 c15,d15 */
        /* trans 4s */
        "vtrn.32    q0, q1\n" /* get q0: a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                 a9,b9, a10,b10, a11,b11; q1: a4,b4, a5,b5,
                                 a6,b6, a7,b7, a12,b12, a13,b13, a14,b14,
                                 a15,b15 */
        "vtrn.32    q2, q3\n" /* get q2: c0,d0, c1,d1, c2,d2, c3,d3, c8,d8,
                                 c9,d9, c10,d10, c11,d11; q3: c4,d4, c5,d5,
                                 c6,d6, c7,d7, c12,d12, c13,d13, c14,d14,
                                 c15,d15 */
        /* trans 2d */
        "vswp   d1, d4\n" /* get q0: a0,b0, a1,b1, a2,b2, a3,b3, c0,d0, c1,d1,
                             c2,d2, c3,d3; q2: a8,b8, a9,b9, a10,b10, a11,b11,
                             c8,d8, c9,d9, c10,d10, c11,d11 */
        "vswp   d3, d6\n" /* get q1: a4,b4, a5,b5, a6,b6, a7,b7, c4,d4, c5,d5,
                             c6,d6, c7,d7; q3: a12,b12, a13,b13, a14,b14,
                             a15,b15, c12,d12, c13,d13, c14,d14, c15,d15 */
        "vst1.8 {d0-d1}, [%[ptr_out]], %[stride]\n" /* write block0, address +
                                                       stride */
        "vst1.8 {d2-d3}, [%[ptr_out]], %[stride]\n" /* write block1, address +
                                                       stride */
        "vst1.8 {d4-d5}, [%[ptr_out]], %[stride]\n" /* write block2, address +
                                                       stride */
        "vst1.8 {d6-d7}, [%[ptr_out]], %[stride]\n" /* write block3, address +
                                                       stride */
        "vld1.8 {d0-d1},    [%[ptr0]]!\n"           /* load r0 */
        "vld1.8 {d2-d3},    [%[ptr1]]!\n"           /* load r1 */
        "vld1.8 {d4-d5},    [%[ptr2]]!\n"           /* load r2 */
        "vld1.8 {d6-d7},    [%[ptr3]]!\n"           /* load r3 */
        "bgt    0b\n"                               /* jump to main loop */
        "1:\n"                                      /* process remain */
        "cmp    %[rem], #0\n"                       /* check remain */
        "beq    2f\n"                               /* skip to remain */
        /* bit select */
        "vbif   q0, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q1, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q2, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q3, %q[vzero], %q[mask]\n" /* pad 0 */
        /* trans 16b */
        "vtrn.8 q0, q1\n" /* get q0: a0,b0, a2,b2, a4,b4, a6,b6, a8,b8, a10,b10,
                             a12,b12, a14,b14; q1: a1,b1, a3,b3, a5,b5, a7,b7,
                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "vtrn.8 q2, q3\n" /* get q2: c0,d0, c2,d2, c4,d4, c6,d6, c8,d8, c10,d10,
                             c12,d12, c14,d14; q3: c0,d0, c2,d2, c4,d4, c6,d6,
                             c8,d8, c10,d10, c12,d12, c14,d14 */
        /* trans 8h */
        "vtrn.16    q0, q1\n" /* get q0: a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                 a9,b9, a12,b12, a13,b13; q1: a2,b2, a3,b3,
                                 a6,b6, a7,b7, a10,b10, a11,b11, a14,b14,
                                 a15,b15 */
        "vtrn.16    q2, q3\n" /* get q2: c0,d0, c1,d1, c4,d4, c5,d5, c8,d8,
                                 c9,d9, c12,d12, c13,d13; q3: c2,d2, c3,d3,
                                 c6,d6, c7,d7, c10,d10, c11,d11, c14,d14,
                                 c15,d15 */
        /* trans 4s */
        "vtrn.32    q0, q1\n" /* get q0: a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                 a9,b9, a10,b10, a11,b11; q1: a4,b4, a5,b5,
                                 a6,b6, a7,b7, a12,b12, a13,b13, a14,b14,
                                 a15,b15 */
        "vtrn.32    q2, q3\n" /* get q2: c0,d0, c1,d1, c2,d2, c3,d3, c8,d8,
                                 c9,d9, c10,d10, c11,d11; q3: c4,d4, c5,d5,
                                 c6,d6, c7,d7, c12,d12, c13,d13, c14,d14,
                                 c15,d15 */
        /* trans 2d */
        "vswp   d1, d4\n" /* get q0: a0,b0, a1,b1, a2,b2, a3,b3, c0,d0, c1,d1,
                             c2,d2, c3,d3; q2: a8,b8, a9,b9, a10,b10, a11,b11,
                             c8,d8, c9,d9, c10,d10, c11,d11 */
        "vswp   d3, d6\n" /* get q1: a4,b4, a5,b5, a6,b6, a7,b7, c4,d4, c5,d5,
                             c6,d6, c7,d7; q3: a12,b12, a13,b13, a14,b14,
                             a15,b15, c12,d12, c13,d13, c14,d14, c15,d15 */
        /* check remain size */
        "subs    %[rem], %[rem], #1\n"              /* check remain num */
        "vst1.8 {d0-d1}, [%[ptr_out]], %[stride]\n" /* write 0 */
        "beq    2f\n"                               /* remain = 1 */
        "subs    %[rem], %[rem], #1\n"              /* check remain num */
        "vst1.8 {d2-d3}, [%[ptr_out]], %[stride]\n" /* write 1 */
        "beq    2f\n"                               /* remain = 2 */
        "subs    %[rem], %[rem], #1\n"              /* check remain num */
        "vst1.8 {d4-d5}, [%[ptr_out]], %[stride]\n" /* write 2 */
        "beq    2f\n"                               /* remain = 3 */
        "vst1.8 {d6-d7}, [%[ptr_out]], %[stride]\n" /* write 3 */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [k] "+r"(k), [rem] "+r"(rem),
          [ptr_out] "+r"(ptr_out)
        : [mask] "w"(vmask), [vzero] "w"(vzero), [stride] "r"(stride_out)
        : "q0", "q1", "q2", "q3", "cc");
#endif  // __aarch64__
    // clang-format on
  }
  LITE_PARALLEL_COMMON_END();
  free(zerobuf);
}

/**************************************************************************/
// for armv8
// prepack B according to gemm kernel
// B block size: (<4x2>x4) x2, can be described as below:
// origin B data:
// B_origin(no trans, k x n):
//      r0: ==>   a0, a1, a2, a3 .... a12, a13, a14, a15
//      r1: ==>   b0, b1, b2, b3 .... b12, b13, b14, b15
//      r2: ==>   c0, c1, c2, c3 .... c12, c13, c14, c15
//      r3: ==>   d0, d1, d2, d3 .... d12, d13, d14, d15
// packed B:
//      a0,b0, a1,b1, a2,b2, a3,b3;
//      c0,d0, c1,d1, c2,d2, c3,d3;
//                   .
//                   .
//                   .
//      a12,b12, a13,b13, a14,b14, a15,b15;
//      c12,d12, c13,d13, c14,d14, c15,d15;
// for armv7
// prepack B according to gemm kernel
// B block size: (<4x2>x4) x2, can be described as below:
// origin B data:
// B_origin(no trans, k x n):
//      r0: ==>   a0, a1, a2, a3, a4, a5, a6, a7
//      r1: ==>   b0, b1, b2, b3, b4, b5, b6, b7
//      r2: ==>   c0, c1, c2, c3, c4, c5, c6, c7
//      r3: ==>   d0, d1, d2, d3, d4, d5, d6, d7
// packed B:
//      a0,b0, a1,b1, a2,b2, a3,b3;
//      a4,b4, a5,b5, a6,b6, a7,b7;
//      c0,d0, c1,d1, c2,d2, c3,d3;
//      c4,d4, c5,d5, c6,d6, c7,d7;
/***************************************************************************/
void packb_int8(int8_t* out,
                const int8_t* in,
                const int ldin,
                const int k0,
                const int kmax,
                const int n0,
                const int nmax,
                const int8_t* zerobuf) {
  const int8_t* inptr = in + k0 * ldin + n0;
  const uint8_t mask_buffer[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int x_len = nmax - n0;
  int y_len = kmax - k0;
  int kup = ROUNDUP(y_len, KBLOCK_INT8);
  int kcnt = x_len / NBLOCK_INT8_OTH;
  int rem = x_len & (NBLOCK_INT8_OTH - 1);
  int stride_out = NBLOCK_INT8_OTH * kup;

  int8x16_t vzero = vdupq_n_s8(0);
  uint8x16_t vmask = vcltq_u8(vld1q_u8(mask_buffer), vdupq_n_u8(rem));
  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, 0, KBLOCK_INT8) {
    const int8_t* ptr0 = inptr + y * ldin;
    const int8_t* ptr1 = ptr0 + ldin;
    const int8_t* ptr2 = ptr1 + ldin;
    const int8_t* ptr3 = ptr2 + ldin;
    if (y + KBLOCK_INT8 > y_len) {
      switch (y + KBLOCK_INT8 - y_len) {
        case 3:
          ptr1 = zerobuf;
        case 2:
          ptr2 = zerobuf;
        case 1:
          ptr3 = zerobuf;
        default:
          break;
      }
    }
    int8_t* outptr_row_col = out + y * NBLOCK_INT8_OTH;
    int k = kcnt;
// clang-format off
#ifdef __aarch64__
    asm volatile(
    "ld1    {v0.16b},   [%[ptr0]],  #16\n" /* load r0 */
    "ld1    {v1.16b},   [%[ptr1]],  #16\n" /* load r1 */
    "ld1    {v2.16b},   [%[ptr2]],  #16\n" /* load r2 */
    "ld1    {v3.16b},   [%[ptr3]],  #16\n" /* load r3 */
    "cbz    %w[k], 1f\n"                   /* jump to remain */
    "0:\n"                                 /* main loop */
    /* trans 16b */
    "trn1   v4.16b, v0.16b, v1.16b\n" /* get a0,b0, a2,b2, a4,b4, a6,b6,
                                         a8,b8, a10,b10, a12,b12, a14,b14 */
    "trn2   v5.16b, v0.16b, v1.16b\n" /* get a1,b1, a3,b3, a5,b5, a7,b7,
                                         a9,b9, a11,b11, a13,b13, a15,b15 */
    "trn1   v6.16b, v2.16b, v3.16b\n" /* get c0,d0, c2,d2, c4,d4, c6,d6,
                                         c8,d8, c10,d10, c12,d12, c14,d14 */
    "trn2   v7.16b, v2.16b, v3.16b\n" /* get c1,d1, c3,d3, c5,d5, c7,d7,
                                         c9,d9, c11,d11, c13,d13, c15,d15 */
    "ld1    {v0.16b},   [%[ptr0]],  #16\n" /* load r0 */
    "ld1    {v1.16b},   [%[ptr1]],  #16\n" /* load r1 */
    "subs   %w[k], %w[k], #1\n"            /* loop cnt -1 */
    /* trans 8h */
    "trn1   v8.8h, v4.8h, v5.8h\n" /* get a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                      a9,b9, a12,b12, a13,b13 */
    "trn2   v9.8h, v4.8h, v5.8h\n" /* get a2,b2, a3,b3, a6,b6, a7,b7,
                                      a10,b10, a11,b11, a14,b14, a15,b15 */
    "trn1   v10.8h, v6.8h, v7.8h\n" /* get c0,d0, c1,d1, c4,d4, c5,d5,
                                       c8,d8, c9,d9, c12,d12, c13,d13 */
    "trn2   v11.8h, v6.8h, v7.8h\n" /* get c2,d2, c3,d3, c6,d6, c7,d7,
                                       c10,d10, c11,d11, c14,d14, c15,d15 */
    /* trans 4s */
    "ld1    {v2.16b},   [%[ptr2]],  #16\n" /* load r2 */
    "trn1   v4.4s, v8.4s, v9.4s\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                      a9,b9, a10,b10, a11,b11 */
    "trn2   v5.4s, v8.4s, v9.4s\n" /* get a4,b4, a5,b5, a6,b6, a7,b7,
                                      a12,b12, a13,b13, a14,b14, a15,b15 */
    "trn1   v6.4s, v10.4s, v11.4s\n" /* get c0,d0, c1,d1, c2,d2, c3,d3,
                                        c8,d8, c9,d9, c10,d10, c11,d11 */
    "trn2   v7.4s, v10.4s, v11.4s\n" /* get c4,d4, c5,d5, c6,d6, c7,d7,
                                        c12,d12, c13,d13, c14,d14, c15,d15
                                        */
    /* trans 2d */
    "ld1    {v3.16b},   [%[ptr3]],  #16\n" /* load r3 */
    "trn1   v8.2d, v4.2d, v6.2d\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, c0,d0,
                                      c1,d1, c2,d2, c3,d3 */
    "trn2   v10.2d, v4.2d, v6.2d\n" /* get a8,b8, a9,b9, a10,b10, a11,b11,
                                       c8,d8, c9,d9, c10,d10, c11,d11 */
    "trn1   v9.2d, v5.2d, v7.2d\n" /* get a4,b4, a5,b5, a6,b6, a7,b7, c4,d4,
                                      c5,d5, c6,d6, c7,d7 */
    "trn2   v11.2d, v5.2d, v7.2d\n" /* get a12,b12, a13,b13, a14,b14,
                                       a15,b15, c12,d12, c13,d13, c14,d14,
                                       c15,d15 */
    "st1    {v8.16b, v9.16b, v10.16b, v11.16b},   [%[ptr_out]], %[stride]\n"
    "bgt    0b\n"          /* jump to main loop */
    "1:\n"                 /* process remain */
    "cbz    %w[rem], 2f\n" /* jump to remain */
    /* bit select */
    "bif    v0.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
    "bif    v1.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
    "bif    v2.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
    "bif    v3.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
    /* trans 16b */
    "trn1   v4.16b, v0.16b, v1.16b\n" /* get a0,b0, a2,b2, a4,b4, a6,b6,
                                         a8,b8, a10,b10, a12,b12, a14,b14 */
    "trn2   v5.16b, v0.16b, v1.16b\n" /* get a1,b1, a3,b3, a5,b5, a7,b7,
                                         a9,b9, a11,b11, a13,b13, a15,b15 */
    "trn1   v6.16b, v2.16b, v3.16b\n" /* get c0,d0, c2,d2, c4,d4, c6,d6,
                                         c8,d8, c10,d10, c12,d12, c14,d14 */
    "trn2   v7.16b, v2.16b, v3.16b\n" /* get c1,d1, c3,d3, c5,d5, c7,d7,
                                         c9,d9, c11,d11, c13,d13, c15,d15 */
    /* trans 8h */
    "trn1   v8.8h, v4.8h, v5.8h\n" /* get a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                      a9,b9, a12,b12, a13,b13 */
    "trn2   v9.8h, v4.8h, v5.8h\n" /* get a2,b2, a3,b3, a6,b6, a7,b7,
                                      a10,b10, a11,b11, a14,b14, a15,b15 */
    "trn1   v10.8h, v6.8h, v7.8h\n" /* get c0,d0, c1,d1, c4,d4, c5,d5,
                                       c8,d8, c9,d9, c12,d12, c13,d13 */
    "trn2   v11.8h, v6.8h, v7.8h\n" /* get c2,d2, c3,d3, c6,d6, c7,d7,
                                       c10,d10, c11,d11, c14,d14, c15,d15 */
    /* trans 4s */
    "trn1   v4.4s, v8.4s, v9.4s\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                      a9,b9, a10,b10, a11,b11 */
    "trn2   v5.4s, v8.4s, v9.4s\n" /* get a4,b4, a5,b5, a6,b6, a7,b7,
                                      a12,b12, a13,b13, a14,b14, a15,b15 */
    "trn1   v6.4s, v10.4s, v11.4s\n" /* get c0,d0, c1,d1, c2,d2, c3,d3,
                                        c8,d8, c9,d9, c10,d10, c11,d11 */
    "trn2   v7.4s, v10.4s, v11.4s\n" /* get c4,d4, c5,d5, c6,d6, c7,d7,
                                        c12,d12, c13,d13, c14,d14, c15,d15
                                        */
    /* trans 2d */
    "trn1   v8.2d, v4.2d, v6.2d\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, c0,d0,
                                      c1,d1, c2,d2, c3,d3 */
    "trn2   v10.2d, v4.2d, v6.2d\n" /* get a8,b8, a9,b9, a10,b10, a11,b11,
                                       c8,d8, c9,d9, c10,d10, c11,d11 */
    "trn1   v9.2d, v5.2d, v7.2d\n" /* get a4,b4, a5,b5, a6,b6, a7,b7, c4,d4,
                                      c5,d5, c6,d6, c7,d7 */
    "trn2   v11.2d, v5.2d, v7.2d\n" /* get a12,b12, a13,b13, a14,b14,
                                       a15,b15, c12,d12, c13,d13, c14,d14,
                                       c15,d15 */
    "st1    {v8.16b, v9.16b, v10.16b, v11.16b},   [%[ptr_out]]\n" /* save to
                                                                     memory
                                                                     */
    /* end */
    "2:\n" /* end */
    : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
      [ptr3] "+r"(ptr3), [k] "+r"(k), [ptr_out] "+r"(outptr_row_col)
    : [rem] "r"(rem), [mask] "w"(vmask), [vzero] "w"(vzero),
      [stride] "r"(stride_out)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
      "v11", "cc");
#else   // armv7
    asm volatile(
        "vld1.8 {d0},   [%[ptr0]]!\n" /* load r0, a0,a1,a2,a3,a4,a5,a6,a7 */
        "vld1.8 {d1},   [%[ptr1]]!\n" /* load r1, b0,b1,b2,b3,b4,b5,b6,b7 */
        "vld1.8 {d2},   [%[ptr2]]!\n" /* load r2, c0,c1,c2,c3,c4,c5,c6,c7 */
        "vld1.8 {d3},   [%[ptr3]]!\n" /* load r3, d0,d1,d2,d3,d4,d5,d6,d7 */
        "cmp    %[k], #0\n"           /* check main loop count */
        "beq    1f\n"                 /* jump to remain */
        "0:\n"                        /* main loop */
        /* trans 8b */
        "vtrn.8 d0, d1\n" /* get d0: a0,b0, a2,b2, a4,b4, a6,b6; d1: a1,b1,
                             a3,b3, a5,b5, a7,b7 */
        "vtrn.8 d2, d3\n" /* get d2: c0,d0, c2,d2, c4,d4, c6,d6; d3: c1,d1,
                             c3,d3, c5,d5, c7,d7 */
        /* trans 4h */
        "vtrn.16    d0, d1\n" /* get d0: a0,b0, a1,b1, a4,b4, a5,b5; d1: a2,b2,
                                 a3,b3, a6,b6, a7,b7 */
        "vtrn.16    d2, d3\n" /* get d2: c0,d0, c1,d1, c4,d4, c5,d5; d3: c2,d2,
                                 c3,d3, c6,d6, c7,d7 */
        "subs   %[k],   %[k],   #1\n" /* loop - 1 */
        /* trans 2s */
        "vtrn.32    d0, d1\n" /* get d0: a0,b0, a1,b1, a2,b2, a3,b3; d1: a4,b4,
                                 a5,b5, a6,b6, a7,b7 */
        "vtrn.32    d2, d3\n" /* get d2: c0,d0, c1,d1, c2,d2, c3,d3; d3: c4,d4,
                                 c5,d5, c6,d6, c7,d7 */
        "vst1.8 {d0-d3},   [%[ptr_out]], %[stride]\n" /* save to memory */
        "vld1.8 {d0},   [%[ptr0]]!\n" /* load r0, a0,a1,a2,a3,a4,a5,a6,a7 */
        "vld1.8 {d1},   [%[ptr1]]!\n" /* load r1, b0,b1,b2,b3,b4,b5,b6,b7 */
        "vld1.8 {d2},   [%[ptr2]]!\n" /* load r2, c0,c1,c2,c3,c4,c5,c6,c7 */
        "vld1.8 {d3},   [%[ptr3]]!\n" /* load r3, d0,d1,d2,d3,d4,d5,d6,d7 */
        "bgt    0b\n"                 /* jump to main loop */
        "1:\n"                        /* process remain */
        "cmp    %[rem], #0\n"         /* check remain size */
        "beq    2f\n"                 /* jump to end */
        /* bit select */
        "vbif    d0, %e[vzero], %e[mask]\n" /* pad 0 */
        "vbif    d1, %e[vzero], %e[mask]\n" /* pad 0 */
        "vbif    d2, %e[vzero], %e[mask]\n" /* pad 0 */
        "vbif    d3, %e[vzero], %e[mask]\n" /* pad 0 */
        /* trans 8b */
        "vtrn.8 d0, d1\n" /* get d0: a0,b0, a2,b2, a4,b4, a6,b6; d1: a1,b1,
                             a3,b3, a5,b5, a7,b7 */
        "vtrn.8 d2, d3\n" /* get d2: c0,d0, c2,d2, c4,d4, c6,d6; d3: c1,d1,
                             c3,d3, c5,d5, c7,d7 */
        /* trans 4h */
        "vtrn.16    d0, d1\n" /* get d0: a0,b0, a1,b1, a4,b4, a5,b5; d1: a2,b2,
                                 a3,b3, a6,b6, a7,b7 */
        "vtrn.16    d2, d3\n" /* get d2: c0,d0, c1,d1, c4,d4, c5,d5; d3: c2,d2,
                                 c3,d3, c6,d6, c7,d7 */
        /* trans 2s */
        "vtrn.32    d0, d1\n" /* get d0: a0,b0, a1,b1, a2,b2, a3,b3; d1: a4,b4,
                                 a5,b5, a6,b6, a7,b7 */
        "vtrn.32    d2, d3\n" /* get d2: c0,d0, c1,d1, c2,d2, c3,d3; d3: c4,d4,
                                 c5,d5, c6,d6, c7,d7 */
        "vst1.8 {d0-d3},   [%[ptr_out]]\n" /* save to memory */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [k] "+r"(k), [ptr_out] "+r"(outptr_row_col)
        : [rem] "r"(rem), [mask] "w"(vmask), [vzero] "w"(vzero),
          [stride] "r"(stride_out)
        : "q0", "q1", "cc");
#endif  // __aarch64__
    // clang-format on
  }
  LITE_PARALLEL_COMMON_END();
}

/************************************************************************/
// prepack B according to gemm kernel
// origin B data:
// B_origin(transpose, n x k:
//      k unroll 2, a0=k0,k1
//      r0: ==>   a0, a1, a2, a3, a4, a5, a6, a7
//      r1: ==>   b0, b1, b2, b3, b4, b5, b6, b7
//      r2: ==>   c0, c1, c2, c3, c4, c5, c6, c7
//      r3: ==>   d0, d1, d2, d3, d4, d5, d6, d7
//      r4: ==>   e0, e1, e2, e3, e4, e5, e6, e7
//      r5: ==>   f0, f1, f2, f3, f4, f5, f6, f7
//      r6: ==>   g0, g1, g2, g3, g4, g5, g6, g7
//      r7: ==>   h0, h1, h2, h3, h4, h5, h6, h7
// for armv8:
// B block size: (<4x2>x4) x2, can be described as below:
// packed B:
//      a0,b0, c0,d0, a1,b1, c1,d1;
//      e0,f0, g0,h0, e1,f1, g1,h1;--block0, address+64
//                   .
//                   .
//                   .
//      a6,b6, c6,d6, a7,b7, c7,d7;
//      e6,f6, g6,h6, e7,f7, g7,h7;--block3, address+64
// for armv7:
// B block size: (<8x2>x1) x2, can be described as below:
// packed B:
//      a0,b0, c0,d0, e0,f0, g0,h0;
//      a1,b1, c1,d1, e1,f1, g1,h1;--block0, address+32
//                   .
//                   .
//                   .
//      a6,b6, c6,d6, e6,f6, g6,h6;
//      a7,b7, c7,d7, e7,f7, g7,h7;--block3, address+32
/*******************************************************************/
void packb_trans_int8(int8_t* out,
                      const int8_t* in,
                      const int ldin,
                      const int k0,
                      const int kmax,
                      const int n0,
                      const int nmax,
                      const int8_t* zerobuf) {
  const int KUNROLL = 4;
  const int NUNROLL = 8;
  const int RATIO = NBLOCK_INT8_OTH / NUNROLL;
  const int8_t* inptr = in + n0 * ldin + k0;
  const uint8_t mask_buffer[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int y_len = nmax - n0;
  int x_len = kmax - k0;
  int yup = ROUNDUP(y_len, NBLOCK_INT8_OTH);
  const int kup = ROUNDUP(x_len, KBLOCK_INT8);
  const int KSTRIDE = KBLOCK_INT8 * KUNROLL;
  int kcnt = x_len / KSTRIDE;
  int x_rem = (x_len & (KSTRIDE - 1));
  int k_rem = (x_rem + KBLOCK_INT8 - 1) / KBLOCK_INT8;
  const int stride_inner = KBLOCK_INT8 * NUNROLL;
  const int stride_outer = kup * NBLOCK_INT8_OTH;
  const int ncnt = yup / NUNROLL;

  int8x16_t vzero = vdupq_n_s8(0);
  uint8x16_t vmask = vcltq_u8(vld1q_u8(mask_buffer), vdupq_n_u8(x_rem));

  LITE_PARALLEL_BEGIN(y, tid, ncnt) {
    int idx = y * NUNROLL;
    const int8_t* ptr0 = inptr + idx * ldin;
    const int8_t* ptr1 = ptr0 + ldin;
    const int8_t* ptr2 = ptr1 + ldin;
    const int8_t* ptr3 = ptr2 + ldin;
    const int8_t* ptr4 = ptr3 + ldin;
    const int8_t* ptr5 = ptr4 + ldin;
    const int8_t* ptr6 = ptr5 + ldin;
    const int8_t* ptr7 = ptr6 + ldin;
    // only for ratio = 0 or 1
    int8_t* ptr_out =
        out + (y & (RATIO - 1)) * stride_inner + (y / RATIO) * stride_outer;
    if (idx + NUNROLL > y_len) {
      switch (idx + NUNROLL - y_len) {
        case 8:
          ptr0 = zerobuf;
        case 7:
          ptr1 = zerobuf;
        case 6:
          ptr2 = zerobuf;
        case 5:
          ptr3 = zerobuf;
        case 4:
          ptr4 = zerobuf;
        case 3:
          ptr5 = zerobuf;
        case 2:
          ptr6 = zerobuf;
        case 1:
          ptr7 = zerobuf;
        default:
          break;
      }
    }
    int k = kcnt;
    int rem = k_rem;
// clang-format off
#ifdef __aarch64__
    asm volatile(
        "cbz    %w[k], 1f\n" /* skip  main loop */
        /* main loop */
        "0:\n"                              /* main loop */
        "ld1    {v0.16b}, [%[ptr0]], #16\n" /* load n0, k0~k15 */
        "ld1    {v1.16b}, [%[ptr1]], #16\n" /* load n1, k0~k15 */
        "ld1    {v2.16b}, [%[ptr2]], #16\n" /* load n2, k0~k15 */
        "ld1    {v3.16b}, [%[ptr3]], #16\n" /* load n3, k0~k15 */
        "ld1    {v4.16b}, [%[ptr4]], #16\n" /* load n4, k0~k15 */
        "ld1    {v5.16b}, [%[ptr5]], #16\n" /* load n5, k0~k15 */
        "ld1    {v6.16b}, [%[ptr6]], #16\n" /* load n6, k0~k15 */
        "ld1    {v7.16b}, [%[ptr7]], #16\n" /* load n7, k0~k15 */
        /* trans, 8h */
        "trn1   v8.8h,  v0.8h,  v1.8h\n" /* trans, zip n0,n1 */
        "trn2   v9.8h,  v0.8h,  v1.8h\n" /* trans, zip n0,n1 */
        "trn1  v10.8h,  v2.8h,  v3.8h\n" /* trans, zip n2,n3 */
        "trn2  v11.8h,  v2.8h,  v3.8h\n" /* trans, zip n2,n3 */
        "trn1  v12.8h,  v4.8h,  v5.8h\n" /* trans, zip n4,n5 */
        "trn2  v13.8h,  v4.8h,  v5.8h\n" /* trans, zip n4,n5 */
        "trn1  v14.8h,  v6.8h,  v7.8h\n" /* trans, zip n6,n7 */
        "trn2  v15.8h,  v6.8h,  v7.8h\n" /* trans, zip n6,n7 */
        /* trans, 4s */
        "trn1  v16.4s,  v8.4s, v10.4s\n" /* trans, block 0 */
        "trn2  v17.4s,  v8.4s, v10.4s\n" /* trans, block 0 */
        "trn1  v18.4s,  v9.4s, v11.4s\n" /* trans, block 0 */
        "trn2  v19.4s,  v9.4s, v11.4s\n" /* trans, block 0 */
        "trn1  v20.4s, v12.4s, v14.4s\n" /* trans, block 1 */
        "trn2  v21.4s, v12.4s, v14.4s\n" /* trans, block 1 */
        "trn1  v22.4s, v13.4s, v15.4s\n" /* trans, block 1 */
        "trn2  v23.4s, v13.4s, v15.4s\n" /* trans, block 1 */
        "subs   %w[k],  %w[k],  #1\n"    /* loop count -1 */
        /* trans, 2d */
        "trn1   v8.2d, v16.2d, v18.2d\n" /* trans, block 0, out0 */
        "trn1   v9.2d, v20.2d, v22.2d\n" /* trans, block 1, out0 */
        "trn1  v10.2d, v17.2d, v19.2d\n" /* trans, block 0, out1 */
        "trn1  v11.2d, v21.2d, v23.2d\n" /* trans, block 1, out1 */
        "trn2  v12.2d, v16.2d, v18.2d\n" /* trans, block 0, out2 */
        "trn2  v13.2d, v20.2d, v22.2d\n" /* trans, block 1, out2 */
        "trn2  v14.2d, v17.2d, v19.2d\n" /* trans, block 0, out3 */
        "trn2  v15.2d, v21.2d, v23.2d\n" /* trans, block 1, out3 */
        /* store result */
        "stp    q8, q9,   [%[ptr_out]],#64\n" /* write 0 */
        "stp  q10, q11,   [%[ptr_out]],#64\n" /* write 1 */
        "stp  q12, q13,   [%[ptr_out]],#64\n" /* write 2 */
        "stp  q14, q15,   [%[ptr_out]],#64\n" /* write 3 */
        "bgt    0b\n"                         /* jump to main loop */
        /* process remain */
        "1:\n"                         /* process remains */
        "cbz    %w[rem], 2f\n"         /* no remain, jump to end */
        "ld1    {v0.16b}, [%[ptr0]]\n" /* load n0, k0~k15 */
        "ld1    {v1.16b}, [%[ptr1]]\n" /* load n1, k0~k15 */
        "ld1    {v2.16b}, [%[ptr2]]\n" /* load n2, k0~k15 */
        "ld1    {v3.16b}, [%[ptr3]]\n" /* load n3, k0~k15 */
        "ld1    {v4.16b}, [%[ptr4]]\n" /* load n4, k0~k15 */
        "ld1    {v5.16b}, [%[ptr5]]\n" /* load n5, k0~k15 */
        "ld1    {v6.16b}, [%[ptr6]]\n" /* load n6, k0~k15 */
        "ld1    {v7.16b}, [%[ptr7]]\n" /* load n7, k0~k15 */
        /* bit select */
        "bif    v0.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v1.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v2.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v3.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v4.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v5.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v6.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v7.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        /* trans, 8h */
        "trn1   v8.8h,  v0.8h,  v1.8h\n" /* trans, zip n0,n1 */
        "trn2   v9.8h,  v0.8h,  v1.8h\n" /* trans, zip n0,n1 */
        "trn1  v10.8h,  v2.8h,  v3.8h\n" /* trans, zip n2,n3 */
        "trn2  v11.8h,  v2.8h,  v3.8h\n" /* trans, zip n2,n3 */
        "trn1  v12.8h,  v4.8h,  v5.8h\n" /* trans, zip n4,n5 */
        "trn2  v13.8h,  v4.8h,  v5.8h\n" /* trans, zip n4,n5 */
        "trn1  v14.8h,  v6.8h,  v7.8h\n" /* trans, zip n6,n7 */
        "trn2  v15.8h,  v6.8h,  v7.8h\n" /* trans, zip n6,n7 */
        /* trans, 4s */
        "trn1  v16.4s,  v8.4s, v10.4s\n" /* trans, block 0 */
        "trn2  v17.4s,  v8.4s, v10.4s\n" /* trans, block 0 */
        "trn1  v18.4s,  v9.4s, v11.4s\n" /* trans, block 0 */
        "trn2  v19.4s,  v9.4s, v11.4s\n" /* trans, block 0 */
        "trn1  v20.4s, v12.4s, v14.4s\n" /* trans, block 1 */
        "trn2  v21.4s, v12.4s, v14.4s\n" /* trans, block 1 */
        "trn1  v22.4s, v13.4s, v15.4s\n" /* trans, block 1 */
        "trn2  v23.4s, v13.4s, v15.4s\n" /* trans, block 1 */
        /* trans, 2d */
        "trn1   v8.2d, v16.2d, v18.2d\n" /* trans, block 0, out0 */
        "trn1   v9.2d, v20.2d, v22.2d\n" /* trans, block 1, out0 */
        "trn1  v10.2d, v17.2d, v19.2d\n" /* trans, block 0, out1 */
        "trn1  v11.2d, v21.2d, v23.2d\n" /* trans, block 1, out1 */
        "trn2  v12.2d, v16.2d, v18.2d\n" /* trans, block 0, out2 */
        "trn2  v13.2d, v20.2d, v22.2d\n" /* trans, block 1, out2 */
        "trn2  v14.2d, v17.2d, v19.2d\n" /* trans, block 0, out3 */
        "trn2  v15.2d, v21.2d, v23.2d\n" /* trans, block 1, out3 */
        /* check remain size */
        "subs    %w[rem], %w[rem], #1\n"      /* check remain num */
        "stp    q8, q9,   [%[ptr_out]],#64\n" /* write 0 */
        "beq    2f\n"                         /* remain = 1 */
        "subs    %w[rem], %w[rem], #1\n"      /* check remain num */
        "stp  q10, q11,   [%[ptr_out]],#64\n" /* write 1 */
        "beq    2f\n"                         /* remain = 2 */
        "subs    %w[rem], %w[rem], #1\n"      /* check remain num */
        "stp  q12, q13,   [%[ptr_out]],#64\n" /* write 2 */
        "beq    2f\n"                         /* remain = 3 */
        "stp  q14, q15,   [%[ptr_out]]\n"     /* write 3 */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [ptr4] "+r"(ptr4), [ptr5] "+r"(ptr5),
          [ptr6] "+r"(ptr6), [ptr7] "+r"(ptr7), [ptr_out] "+r"(ptr_out),
          [k] "+r"(k), [rem] "+r"(rem)
        : [mask] "w"(vmask), [vzero] "w"(vzero)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23", "cc");
#else   // armv7
    asm volatile(
        "cmp    %[k], #0\n" /* check  main loop */
        "beq    1f\n"       /* skip  main loop */
        /* main loop */
        "0:\n"                           /* main loop */
        "vld1.8 {d0-d1}, [%[ptr0]]!\n"   /* load n0, a0~a7 */
        "vld1.8 {d2-d3}, [%[ptr1]]!\n"   /* load n1, b0~b7 */
        "vld1.8 {d4-d5}, [%[ptr2]]!\n"   /* load n2, c0~c7 */
        "vld1.8 {d6-d7}, [%[ptr3]]!\n"   /* load n3, d0~d7 */
        "vld1.8 {d8-d9}, [%[ptr4]]!\n"   /* load n4, e0~e7 */
        "vld1.8 {d10-d11}, [%[ptr5]]!\n" /* load n5, f0~f7 */
        "vld1.8 {d12-d13}, [%[ptr6]]!\n" /* load n6, g0~g7 */
        "vld1.8 {d14-d15}, [%[ptr7]]!\n" /* load n7, h0~h7 */
        /* trans, 8h */
        "vtrn.16    q0, q1\n" /* trans, zip n0,n1, q0: a0b0,a2b2, a4b4,a6b6, q1:
                                 a1b1,a3b3, a5b5,a7b7 */
        "vtrn.16    q2, q3\n" /* trans, zip n2,n3, q2: c0d0,c2d2, c4d4,c6d6, q3:
                                 c1d1,c3d3, c5d5,c7d7 */
        "vtrn.16    q4, q5\n" /* trans, zip n4,n5, q4: e0f0,e2f2, e4f4,e6f6, q5:
                                 e1f1,e3f3, e5f5,e7f7 */
        "vtrn.16    q6, q7\n" /* trans, zip n6,n7, q6: g0h0,g2h2, g4h4,g6h6, q7:
                                 g1h1,g3h3, g5h5,g7h7 */
        /* trans, 4s */
        "vtrn.32    q0, q2\n" /* trans, q0: a0b0,c0d0, a4b4,c4d4, q2: a2b2,c2d2,
                                 a6b6,c6d6 */
        "vtrn.32    q1, q3\n" /* trans, q1: a1b1,c1d1, a5b5,c5d5, q3: a3b3,c3d3,
                                 a7b7,c7d7 */
        "vtrn.32    q4, q6\n" /* trans, q4: e0f0,g0h0, e4f4,g4h4, q6: e2f2,g2h2,
                                 e6f6,g6h6 */
        "vtrn.32    q5, q7\n" /* trans, q5: e1f1,g1h1, e5f5,g5h5, q7: e3f3,g3h3,
                                 e7f7,g7h7 */
        "subs   %[k],  %[k],  #1\n" /* loop count -1 */
        /* trans, 2d */
        "vswp   d1, d8\n"  /* q0: a0b0,c0d0, e0f0,g0h0, q4: a4b4,c4d4, e4f4,g4h4
                              */
        "vswp   d3, d10\n" /* q1: a1b1,c1d1, e1f1,g1h1, q5: a5b5,c5d5, e5f5,g5h5
                              */
        "vswp   d5, d12\n" /* q2: a2b2,c2d2, e2f2,g2h2, q6: a6b6,c6d6, e6f6,g6h6
                              */
        "vswp   d7, d14\n" /* q3: a3b3,c3d3, e3f3,g3h3, q7: a7b7,c7d7, e7f7,g7h7
                              */
        /* store result */
        "vst1.8 {d0-d3},    [%[ptr_out]]!\n" /* write 0 */
        "vst1.8 {d4-d7},    [%[ptr_out]]!\n" /* write 1 */
        "vst1.8 {d8-d11},   [%[ptr_out]]!\n" /* write 2 */
        "vst1.8 {d12-d15},  [%[ptr_out]]!\n" /* write 3 */
        "bgt    0b\n"                        /* jump to main loop */
        /* process remain */
        "1:\n"                           /* process remains */
        "cmp    %[rem], #0\n"            /* check remain */
        "beq    2f\n"                    /* no remain, jump to end */
        "vld1.8 {d0-d1}, [%[ptr0]]!\n"   /* load n0, a0~a7 */
        "vld1.8 {d2-d3}, [%[ptr1]]!\n"   /* load n1, b0~b7 */
        "vld1.8 {d4-d5}, [%[ptr2]]!\n"   /* load n2, c0~c7 */
        "vld1.8 {d6-d7}, [%[ptr3]]!\n"   /* load n3, d0~d7 */
        "vld1.8 {d8-d9}, [%[ptr4]]!\n"   /* load n4, e0~e7 */
        "vld1.8 {d10-d11}, [%[ptr5]]!\n" /* load n5, f0~f7 */
        "vld1.8 {d12-d13}, [%[ptr6]]!\n" /* load n6, g0~g7 */
        "vld1.8 {d14-d15}, [%[ptr7]]!\n" /* load n7, h0~h7 */
        /* bit select */
        "vbif   q0, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q1, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q2, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q3, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q4, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q5, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q6, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q7, %q[vzero], %q[mask]\n" /* pad 0 */
        /* trans, 8h */
        "vtrn.16    q0, q1\n" /* trans, zip n0,n1, q0: a0b0,a2b2, a4b4,a6b6, q1:
                                 a1b1,a3b3, a5b5,a7b7 */
        "vtrn.16    q2, q3\n" /* trans, zip n2,n3, q2: c0d0,c2d2, c4d4,c6d6, q3:
                                 c1d1,c3d3, c5d5,c7d7 */
        "vtrn.16    q4, q5\n" /* trans, zip n4,n5, q4: e0f0,e2f2, e4f4,e6f6, q5:
                                 e1f1,e3f3, e5f5,e7f7 */
        "vtrn.16    q6, q7\n" /* trans, zip n6,n7, q6: g0h0,g2h2, g4h4,g6h6, q7:
                                 g1h1,g3h3, g5h5,g7h7 */
        /* trans, 4s */
        "vtrn.32    q0, q2\n" /* trans, q0: a0b0,c0d0, a4b4,c4d4, q2: a2b2,c2d2,
                                 a6b6,c6d6 */
        "vtrn.32    q1, q3\n" /* trans, q1: a1b1,c1d1, a5b5,c5d5, q3: a3b3,c3d3,
                                 a7b7,c7d7 */
        "vtrn.32    q4, q6\n" /* trans, q4: e0f0,g0h0, e4f4,g4h4, q6: e2f2,g2h2,
                                 e6f6,g6h6 */
        "vtrn.32    q5, q7\n" /* trans, q5: e1f1,g1h1, e5f5,g5h5, q7: e3f3,g3h3,
                                 e7f7,g7h7 */
        /* trans, 2d */
        "vswp   d1, d8\n"  /* q0: a0b0,c0d0, e0f0,g0h0, q4: a4b4,c4d4, e4f4,g4h4
                              */
        "vswp   d3, d10\n" /* q1: a1b1,c1d1, e1f1,g1h1, q5: a5b5,c5d5, e5f5,g5h5
                              */
        "vswp   d5, d12\n" /* q2: a2b2,c2d2, e2f2,g2h2, q6: a6b6,c6d6, e6f6,g6h6
                              */
        "vswp   d7, d14\n" /* q3: a3b3,c3d3, e3f3,g3h3, q7: a7b7,c7d7, e7f7,g7h7
                              */
        /* check remain size */
        "subs    %[rem], %[rem], #1\n"       /* check remain num */
        "vst1.8 {d0-d3},    [%[ptr_out]]!\n" /* write 0 */
        "beq    2f\n"                        /* remain = 1 */
        "subs    %[rem], %[rem], #1\n"       /* check remain num */
        "vst1.8 {d4-d7},    [%[ptr_out]]!\n" /* write 1 */
        "beq    2f\n"                        /* remain = 2 */
        "subs    %[rem], %[rem], #1\n"       /* check remain num */
        "vst1.8 {d8-d11},   [%[ptr_out]]!\n" /* write 2 */
        "beq    2f\n"                        /* remain = 3 */
        "vst1.8 {d12-d15},  [%[ptr_out]]!\n" /* write 3 */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [ptr4] "+r"(ptr4), [ptr5] "+r"(ptr5),
          [ptr6] "+r"(ptr6), [ptr7] "+r"(ptr7), [ptr_out] "+r"(ptr_out),
          [k] "+r"(k), [rem] "+r"(rem)
        : [mask] "w"(vmask), [vzero] "w"(vzero)
        : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "cc");
#endif  // __aarch64__
    // clang-format on
  }
  LITE_PARALLEL_END();
}

#ifdef WITH_ARM_DOTPROD
#ifdef __aarch64__

template <typename Dtype>
void gemm_prepack_sdot_int8(const int8_t* A_packed,
                            const int8_t* B,
                            const float* bias,
                            Dtype* C,
                            int M,
                            int N,
                            int K,
                            bool is_bias,
                            int is_relu,
                            bool is_transB,
                            const float* scale,
                            const float* alpha,
                            ARMContext* ctx) {
  size_t llc_size = ctx->llc_size() / 4;
  auto workspace = ctx->workspace_data<int8_t>();
  //! MBLOCK_INT8_DOT * x (result) + MBLOCK_INT8_DOT * k (A) + x * k (B) = l2
  int x_block = (llc_size - (MBLOCK_INT8_DOT * K)) /
                (sizeof(int8_t) * (K + MBLOCK_INT8_DOT));
  x_block /= NBLOCK_INT8_DOT;
  x_block *= NBLOCK_INT8_DOT;

  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK_INT8_DOT - 1) / NBLOCK_INT8_DOT;
  x_block *= NBLOCK_INT8_DOT;
  x_block = x_block < NBLOCK_INT8_DOT ? NBLOCK_INT8_DOT : x_block;

  int kup = ROUNDUP(K, KBLOCK_INT8);
  // unroll 2 loop
  int tail_pre = ((kup / 4) & (KBLOCK_INT8 - 1));
  int k_pre = (((kup / 4) + KBLOCK_INT8 - 1) / KBLOCK_INT8) - 1;

  bool flag_p_remain = false;
  int remain = 0;
  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    xmax = (xmax > N) ? N : xmax;
    int bblocks = (xmax - x0 + NBLOCK_INT8_DOT - 1) / NBLOCK_INT8_DOT;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK_INT8_DOT;
    if (remain == 12) {
      remain = 0;
      bblocks++;
    }
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    auto b_pannel = static_cast<int8_t*>(workspace);
    if (!is_transB) {
      // K * N
      packb_sdot_int8_n12_n8_n4(b_pannel, B, N, 0, K, x0, xmax);
    } else {
      // N X K
      packb_sdot_int8_n12_n8_n4_trans(b_pannel, B, K, 0, K, x0, xmax);
    }

    LITE_PARALLEL_COMMON_BEGIN(y, tid, M, 0, MBLOCK_INT8_DOT) {
      unsigned int ymax = y + MBLOCK_INT8_DOT;
      ymax = (ymax > M) ? M : ymax;
      float32_t bias_local[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (is_bias) {
        int j = 0;
        for (int i = y; i < ymax && j < 8; i++, j++) {
          bias_local[j] = bias[i];
        }
      }
      float32_t scale_local[8];
      if (scale) {
        int j = 0;
        for (int i = y; i < ymax && j < 8; i++, j++) {
          scale_local[j] = scale[i];
        }
      }
      Dtype cout0[NBLOCK_INT8_DOT];
      Dtype cout1[NBLOCK_INT8_DOT];
      Dtype cout2[NBLOCK_INT8_DOT];
      Dtype cout3[NBLOCK_INT8_DOT];
      Dtype cout4[NBLOCK_INT8_DOT];
      Dtype cout5[NBLOCK_INT8_DOT];
      Dtype cout6[NBLOCK_INT8_DOT];
      Dtype cout7[NBLOCK_INT8_DOT + 16];

      Dtype* c_ptr0 = C + y * N + x0;
      Dtype* c_ptr1 = c_ptr0 + N;
      Dtype* c_ptr2 = c_ptr1 + N;
      Dtype* c_ptr3 = c_ptr2 + N;
      Dtype* c_ptr4 = c_ptr3 + N;
      Dtype* c_ptr5 = c_ptr4 + N;
      Dtype* c_ptr6 = c_ptr5 + N;
      Dtype* c_ptr7 = c_ptr6 + N;

      Dtype* pout0 = cout0;
      Dtype* pout1 = cout1;
      Dtype* pout2 = cout2;
      Dtype* pout3 = cout3;
      Dtype* pout4 = cout4;
      Dtype* pout5 = cout5;
      Dtype* pout6 = cout6;
      Dtype* pout7 = cout7;
      if ((y + 7) >= ymax) {
        switch ((y + 7) - ymax) {
          case 6:
            c_ptr1 = cout1;
          case 5:
            c_ptr2 = cout2;
          case 4:
            c_ptr3 = cout3;
          case 3:
            c_ptr4 = cout4;
          case 2:
            c_ptr5 = cout5;
          case 1:
            c_ptr6 = cout6;
          case 0:
            c_ptr7 = cout7;
          default:
            break;
        }
      }

      // const int8_t *a_ptr_l = A_packed + y * K;
      const int8_t* a_ptr_l = A_packed + y * kup;
      const int8_t* b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks - 1; xb++) {
        if ((y + 7) >= ymax) {
          switch ((y + 7) - ymax) {
            case 6:
              c_ptr1 = cout1;
            case 5:
              c_ptr2 = cout2;
            case 4:
              c_ptr3 = cout3;
            case 3:
              c_ptr4 = cout4;
            case 2:
              c_ptr5 = cout5;
            case 1:
              c_ptr6 = cout6;
            case 0:
              c_ptr7 = cout7;
            default:
              break;
          }
        }
        const int8_t* a_ptr = a_ptr_l;
        int tail = tail_pre;
        int k = k_pre;
        gemm_sdot_int8_kernel<Dtype>(a_ptr,
                                     b_ptr,
                                     bias_local,
                                     c_ptr0,
                                     c_ptr1,
                                     c_ptr2,
                                     c_ptr3,
                                     c_ptr4,
                                     c_ptr5,
                                     c_ptr6,
                                     c_ptr7,
                                     scale_local,
                                     alpha,
                                     is_relu,
                                     k,
                                     tail);
      }
      int remain_a = remain;
      if (remain_a >= 8) {
        const int8_t* a_ptr = a_ptr_l;
        int k = kup / 4;
        gemm_sdot_int8_kernel_8x8<Dtype>(a_ptr,
                                         b_ptr,
                                         bias_local,
                                         c_ptr0,
                                         c_ptr1,
                                         c_ptr2,
                                         c_ptr3,
                                         c_ptr4,
                                         c_ptr5,
                                         c_ptr6,
                                         c_ptr7,
                                         scale_local,
                                         alpha,
                                         is_relu,
                                         k,
                                         0);
        remain_a -= 8;
      }
      if (remain_a >= 4) {
        const int8_t* a_ptr = a_ptr_l;
        int k = kup / 4;
        gemm_sdot_int8_kernel_8x4<Dtype>(a_ptr,
                                         b_ptr,
                                         bias_local,
                                         c_ptr0,
                                         c_ptr1,
                                         c_ptr2,
                                         c_ptr3,
                                         c_ptr4,
                                         c_ptr5,
                                         c_ptr6,
                                         c_ptr7,
                                         scale_local,
                                         alpha,
                                         is_relu,
                                         k,
                                         0);
        remain_a -= 4;
      }
      if (remain_a) {
        const int8_t* a_ptr = a_ptr_l;
        int k = kup / 4;
        gemm_sdot_int8_kernel_8x4<Dtype>(a_ptr,
                                         b_ptr,
                                         bias_local,
                                         pout0,
                                         pout1,
                                         pout2,
                                         pout3,
                                         pout4,
                                         pout5,
                                         pout6,
                                         pout7,
                                         scale_local,
                                         alpha,
                                         is_relu,
                                         k,
                                         0);
        for (int i = 0; i < remain_a; ++i) {
          *c_ptr0++ = cout0[i];
          *c_ptr1++ = cout1[i];
          *c_ptr2++ = cout2[i];
          *c_ptr3++ = cout3[i];
          *c_ptr4++ = cout4[i];
          *c_ptr5++ = cout5[i];
          *c_ptr6++ = cout6[i];
          *c_ptr7++ = cout7[i];
        }
      }
    }
    LITE_PARALLEL_COMMON_END();
  }
}

void prepackA_m8k4_int8(int8_t* out,
                        const int8_t* in,
                        const int ldin,
                        const int m0,
                        const int mmax,
                        const int k0,
                        const int kmax) {
  int x_len = (kmax - k0);
  int8_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * x_len);

  int8_t* dout = out;
  const int8_t* inptr = in;
  int kup = ROUNDUP(x_len, KBLOCK_INT8);
  int stride = kup * 8;
  int remain = x_len % 4;

  LITE_PARALLEL_COMMON_BEGIN(y, tid, mmax, m0, 8) {
    int8_t* outptr = dout + stride * (y - m0) / 8;
    const int8_t* inptr_row[8];
    inptr_row[0] = inptr + y * ldin + k0;
    for (int i = 1; i < 8; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }
    //! cope with row index exceed real size, set to zero buffer
    if ((y + 7) >= mmax) {
      switch ((y + 7) - mmax) {
        case 6:
          inptr_row[1] = zerobuff;
        case 5:
          inptr_row[2] = zerobuff;
        case 4:
          inptr_row[3] = zerobuff;
        case 3:
          inptr_row[4] = zerobuff;
        case 2:
          inptr_row[5] = zerobuff;
        case 1:
          inptr_row[6] = zerobuff;
        case 0:
          inptr_row[7] = zerobuff;
        default:
          break;
      }
    }
    // clang-format off
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        "prfm   pldl1keep, [%[ptr4]]        \n"
        "prfm   pldl1keep, [%[ptr4], #64]   \n"
        "prfm   pldl1keep, [%[ptr5]]        \n"
        "prfm   pldl1keep, [%[ptr5], #64]   \n"
        "prfm   pldl1keep, [%[ptr6]]        \n"
        "prfm   pldl1keep, [%[ptr6], #64]   \n"
        "prfm   pldl1keep, [%[ptr7]]        \n"
        "prfm   pldl1keep, [%[ptr7], #64]   \n"
        :
        :[ptr0] "r"(inptr_row[0]),[ptr1] "r"(inptr_row[1]),[ptr2] "r"(inptr_row[2]),[ptr3] "r"(inptr_row[3]),\
                [ptr4] "r"(inptr_row[4]),[ptr5] "r"(inptr_row[5]),[ptr6] "r"(inptr_row[6]),[ptr7] "r"(inptr_row[7])
        :"memory"
        );

        int x = x_len;

        for (; x > 7; x -= 8) {
            asm volatile(
            "ld1 {v0.8b}, [%[inptr0]], #8 \n" // v0=a0a1a2a3a4a5a6a7
            "ld1 {v1.8b}, [%[inptr1]], #8 \n" // v1=b0b1b2b3b4b5b6b7
            "ld1 {v2.8b}, [%[inptr2]], #8 \n" // v2=c0c1c2c3c4c5c6c7
            "ld1 {v3.8b}, [%[inptr3]], #8 \n" // v3=d0d1d2d3d4d5d6d7

            "ld1 {v4.8b}, [%[inptr4]], #8 \n" // v0=e0e1a2a3a4a5a6a7
            "ld1 {v5.8b}, [%[inptr5]], #8 \n" // v1=f0f1b2b3b4b5b6b7
            "ld1 {v6.8b}, [%[inptr6]], #8 \n" // v2=g0g1c2c3c4c5c6c7
            "ld1 {v7.8b}, [%[inptr7]], #8 \n" // v3=h0h1d2d3d4d5d6d7

            "trn1 v8.2s, v0.2s, v1.2s \n" // v0=a0a1a2a3b0b1b2b3
            "trn2 v9.2s, v0.2s, v1.2s \n" // v0=a4a5a6a7b4b5b6b7
            "trn1 v10.2s, v2.2s, v3.2s \n" // v0=c0c1c2c3d0d1d2d3
            "trn2 v11.2s, v2.2s, v3.2s \n" // v0=c4c5c6c7d4d5d6d7

            "trn1 v12.2s, v4.2s, v5.2s \n" // v0=e0e1e2e3f0f1f2f3
            "trn2 v13.2s, v4.2s, v5.2s \n" // v0=e4e5e6e7f4f5f6f7
            "trn1 v14.2s, v6.2s, v7.2s \n" // v0=g0g1g2g3h0h1h2h3
            "trn2 v15.2s, v6.2s, v7.2s \n" // v0=g4g5g6g7h4h5h6h7

            "st1 {v8.2s}, [%[outptr]], #8\n"
            "st1 {v10.2s}, [%[outptr]], #8\n"
            "st1 {v12.2s}, [%[outptr]], #8\n"
            "st1 {v14.2s}, [%[outptr]], #8\n"

            "st1 {v9.2s}, [%[outptr]], #8\n"
            "st1 {v11.2s}, [%[outptr]], #8\n"
            "st1 {v13.2s}, [%[outptr]], #8\n"
            "st1 {v15.2s}, [%[outptr]], #8\n"

            :[inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
            [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
            [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
            [inptr6] "+r"(inptr_row[6]), [inptr7] "+r"(inptr_row[7]),
            [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                    "v13", "v14", "v15", "v16", "cc", "memory"
            );
        }
        if (x >= 4) {
            asm volatile(
            "mov x1, #4 \n"
            "ld1 {v0.8b}, [%[inptr0]], x1 \n" // v0=a0a1a2a3a4a5a6a7
            "ld1 {v1.8b}, [%[inptr1]], x1 \n" // v1=b0b1b2b3b4b5b6b7
            "ld1 {v2.8b}, [%[inptr2]], x1 \n" // v2=c0c1c2c3c4c5c6c7
            "ld1 {v3.8b}, [%[inptr3]], x1 \n" // v3=d0d1d2d3d4d5d6d7

            "ld1 {v4.8b}, [%[inptr4]], x1 \n" // v0=e0e1a2a3a4a5a6a7
            "ld1 {v5.8b}, [%[inptr5]], x1 \n" // v1=f0f1b2b3b4b5b6b7
            "ld1 {v6.8b}, [%[inptr6]], x1 \n" // v2=g0g1c2c3c4c5c6c7
            "ld1 {v7.8b}, [%[inptr7]], x1 \n" // v3=h0h1d2d3d4d5d6d7

            "trn1 v8.2s, v0.2s, v1.2s \n" // v0=a0a1a2a3b0b1b2b3
            "trn1 v10.2s, v2.2s, v3.2s \n" // v0=c0c1c2c3d0d1d2d3

            "trn1 v12.2s, v4.2s, v5.2s \n" // v0=e0e1e2e3f0f1f2f3
            "trn1 v14.2s, v6.2s, v7.2s \n" // v0=g0g1g2g3h0h1h2h3

            "st1 {v8.2s}, [%[outptr]], #8\n"
            "st1 {v10.2s}, [%[outptr]], #8\n"

            "st1 {v12.2s}, [%[outptr]], #8\n"
            "st1 {v14.2s}, [%[outptr]], #8\n"

            :[inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
            [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
            [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
            [inptr6] "+r"(inptr_row[6]), [inptr7] "+r"(inptr_row[7]),
            [outptr] "+r"(outptr)
            :
            : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                    "v13", "v14", "v15", "v16", "cc", "memory"
            );
            x -= 4;
        }
    // clang-format on
    if (x > 0) {
      for (int i = 0; i < 8; i++) {
        for (int j = x; j > 0; j--) {
          *outptr++ = *inptr_row[i]++;
        }
        for (int j = 0; j < 4 - remain; j++) {
          *outptr++ = 0;
        }
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
}

void prepackA_m8k4_trans_int8(int8_t* out,
                              const int8_t* in,
                              const int ldin,
                              const int m0,
                              const int mmax,
                              const int k0,
                              const int kmax) {
  int8_t* outptr = out;
  const int8_t* inptr = in + k0 * ldin + m0;
  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int right_remain = x_len % 8;
  int kup = ROUNDUP(y_len, KBLOCK_INT8);

  int stride_out = 8 * kup;
  int8_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * x_len);

  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, 0, 4) {
    const int8_t* inptr0 = inptr + y * ldin;
    const int8_t* inptr1 = inptr0 + ldin;
    const int8_t* inptr2 = inptr1 + ldin;
    const int8_t* inptr3 = inptr2 + ldin;

    if (y + 4 > y_len) {
      switch (y + 4 - y_len) {
        case 3:
          inptr1 = zerobuff;
        case 2:
          inptr2 = zerobuff;
        case 1:
          inptr3 = zerobuff;
        default:
          break;
      }
    }
    // clang-format off
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        :
        :[ptr0] "r"(inptr0),[ptr1] "r"(inptr1),[ptr2] "r"(inptr2),
        [ptr3] "r"(inptr3)
        :"memory"
        );

        int8_t *outptr_row = outptr + y * 8;
        int x = 0;
        for (; x < x_len - 7; x += 8) {
            int8_t *out0 = outptr_row;
            asm volatile (
            "ld1 {v0.8b}, [%[inptr0]], #8 \n" // v0 = a0a1a2a3a4a5a6a7
            "ld1 {v1.8b}, [%[inptr1]], #8 \n" // v0 = b0b1b2b3b4b5b6b7
            "ld1 {v2.8b}, [%[inptr2]], #8 \n" // v0 = c0c1c2c3c4c5c6c7
            "ld1 {v3.8b}, [%[inptr3]], #8 \n" // v0 = d0d1d2d3d4d5d6d7

            "trn1 v4.8b, v0.8b, v1.8b \n" // v4 = a0b0a2b2a4b4a6b6
            "trn2 v5.8b, v0.8b, v1.8b \n" // v4 = a1b1a3b3a5b5a7b7
            "trn1 v6.8b, v2.8b, v3.8b \n" // v4 = c0d0c2d2a4b4a6b6
            "trn2 v7.8b, v2.8b, v3.8b \n" // v4 = c1d1c3d3a5b5a7b7

            "trn1 v0.4h, v4.4h, v6.4h \n" // v4 = a0b0c0d0a4b4c4d4
            "trn2 v1.4h, v4.4h, v6.4h \n" // v4 = a2b2c2d2a6b6c6d6
            "trn1 v2.4h, v5.4h, v7.4h \n" // v4 = a1b1c1d1a5b5c5d5
            "trn2 v3.4h, v5.4h, v7.4h \n" // v4 = a3b3c3d3a7b7c7d7

            "trn1 v4.2s, v0.2s, v2.2s \n" //v4 =a0b0c0d0a1b1c1d1
            "trn2 v5.2s, v0.2s, v2.2s \n" //v4 =a4b4c4d4a5b5c5d5
            "trn1 v6.2s, v1.2s, v3.2s \n" //v4 =a2b2c2d2a3b3c3d3
            "trn2 v7.2s, v1.2s, v3.2s \n" //v4 =a6b6c6d6a7b7c7d7

            "st1 {v4.2s}, [%[outr]], #8\n"
            "st1 {v6.2s}, [%[outr]], #8\n"
            "st1 {v5.2s}, [%[outr]], #8\n"
            "st1 {v7.2s}, [%[outr]], #8\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outr] "+r"(out0)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
              "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "cc", "memory"
            );
            outptr_row += stride_out;
        }
    // clang-format on
    if (right_remain > 0) {
      int8_t* out0 = outptr_row;
      for (; x < x_len; x++) {
        *out0++ = *inptr0++;
        *out0++ = *inptr1++;
        *out0++ = *inptr2++;
        *out0++ = *inptr3++;
      }
      for (int i = 0; i < 8 - right_remain; i++) {
        *out0++ = 0;
        *out0++ = 0;
        *out0++ = 0;
        *out0++ = 0;
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
}

void packb_sdot_int8_n12_n8_n4(int8_t* out,
                               const int8_t* in,
                               const int ldin,
                               const int k0,
                               const int kmax,
                               const int n0,
                               const int nmax) {
  int y_len = kmax - k0;
  int x_len = nmax - n0;
  int kup = ROUNDUP(y_len, KBLOCK_INT8);  //  4k
  int8_t zerobuff[x_len];                 // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * x_len);
  int8_t* outptr = out;

  int stride_out = 12 * kup;
  int x = 0;
  int8_t* out0 = outptr;
  const int8_t* inptr = in;
  for (x = 0; x < x_len - 11; x += 12) {
    inptr = in + k0 * ldin + n0 + x;
    // clang-format off
    for (int y = 0; y < y_len; y += 4) {
      // cope with row index exceed real size, set to zero
      const int8_t* inptr0 = inptr + y * ldin;
      const int8_t* inptr1 = inptr0 + ldin;
      const int8_t* inptr2 = inptr1 + ldin;
      const int8_t* inptr3 = inptr2 + ldin;
      asm volatile(
      "prfm   pldl1keep, [%[inptr0]]                \n"
      "prfm   pldl1keep, [%[inptr1]]        \n"
      "prfm   pldl1keep, [%[inptr2]]        \n"
      "prfm   pldl1keep, [%[inptr3]]        \n"
      :
      :[inptr0] "r"(inptr0), [inptr1] "r"(inptr1),
        [inptr2] "r"(inptr2), [inptr3] "r"(inptr3)
      :"memory"
      );
      if (y + 4 > y_len) {
        switch (y + 4 - y_len) {
          case 3:
            inptr1 = zerobuff;
          case 2:
            inptr2 = zerobuff;
          case 1:
            inptr3 = zerobuff;
          default:
            break;
        }
      }
      asm volatile (
      "mov x1, #4 \n"
      "ld1 {v0.8b}, [%[inptr0]], #8 \n" // v0 = a0a1a2a3a4a5a6a7
      "ld1 {v1.8b}, [%[inptr1]], #8 \n" // v0 = b0b1b2b3b4b5b6b7
      "ld1 {v2.8b}, [%[inptr2]], #8 \n" // v0 = c0c1c2c3c4c5c6c7
      "ld1 {v3.8b}, [%[inptr3]], #8 \n" // v0 = d0d1d2d3d4d5d6d7

      "ld1 {v8.8b}, [%[inptr0]] \n" // v0 = a8a9a10a11
      "ld1 {v9.8b}, [%[inptr1]] \n" // v0 = b8b9b10b11
      "ld1 {v10.8b}, [%[inptr2]] \n" // v0 = c8c9c10c11
      "ld1 {v11.8b}, [%[inptr3]] \n" // v0 = d8d9d10d11

      "trn1 v4.8b, v0.8b, v1.8b \n" // v4 = a0b0a2b2a4b4a6b6
      "trn2 v5.8b, v0.8b, v1.8b \n" // v4 = a1b1a3b3a5b5a7b7
      "trn1 v6.8b, v2.8b, v3.8b \n" // v4 = c0d0c2d2a4b4a6b6
      "trn2 v7.8b, v2.8b, v3.8b \n" // v4 = c1d1c3d3a5b5a7b7

      "trn1 v12.8b, v8.8b, v9.8b \n" // v4 = a8b8a10b10a4b4a6b6
      "trn2 v13.8b, v8.8b, v9.8b \n" // v4 = a9b9a11b11a5b5a7b7
      "trn1 v14.8b, v10.8b, v11.8b \n" // v4 = c8d8c10d10a4b4a6b6
      "trn2 v15.8b, v10.8b, v11.8b \n" // v4 = c9d9c11d11a5b5a7b7

      "trn1 v0.4h, v4.4h, v6.4h \n" // v4 = a0b0c0d0a4b4c4d4
      "trn2 v1.4h, v4.4h, v6.4h \n" // v4 = a2b2c2d2a6b6c6d6
      "trn1 v2.4h, v5.4h, v7.4h \n" // v4 = a1b1c1d1a5b5c5d5
      "trn2 v3.4h, v5.4h, v7.4h \n" // v4 = a3b3c3d3a7b7c7d7

      "trn1 v8.4h, v12.4h, v14.4h \n" // v4 = a8b8c8d8
      "trn2 v9.4h, v12.4h, v14.4h \n" // v4 = a10b10c10d10
      "trn1 v10.4h, v13.4h, v15.4h \n" // v4 = a9b9c9d9
      "trn2 v11.4h, v13.4h, v15.4h \n" // v4 = a11b11c11d11

      "trn1 v4.2s, v0.2s, v2.2s \n" //v4 =a0b0c0d0a1b1c1d1
      "trn2 v5.2s, v0.2s, v2.2s \n" //v4 =a4b4c4d4a5b5c5d5
      "trn1 v6.2s, v1.2s, v3.2s \n" //v4 =a2b2c2d2a3b3c3d3
      "trn2 v7.2s, v1.2s, v3.2s \n" //v4 =a6b6c6d6a7b7c7d7

      "trn1 v0.2s, v8.2s, v10.2s \n" //v4 =a8b8c8d8a9b9c9d9
      "trn1 v1.2s, v9.2s, v11.2s \n" //v4 =a10b10c10d10a11b11c11d11

      "st1 {v4.2s}, [%[outr]], #8\n"
      "st1 {v6.2s}, [%[outr]], #8\n"
      "st1 {v5.2s}, [%[outr]], #8\n"
      "st1 {v7.2s}, [%[outr]], #8\n"
      "st1 {v0.2s}, [%[outr]], #8\n"
      "st1 {v1.2s}, [%[outr]], #8\n"
      : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
        [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
        [outr] "+r"(out0)
      :
      : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "cc", "memory"
      );
    }
  }
  if ((x_len - x) >= 8) {
    for (int y = 0; y < y_len; y += 4) {
      inptr = in + k0 * ldin + n0 + x;
      const int8_t* inptr0 = inptr + y * ldin;
      const int8_t* inptr1 = inptr0 + ldin;
      const int8_t* inptr2 = inptr1 + ldin;
      const int8_t* inptr3 = inptr2 + ldin;
      asm volatile(
      "ld1 {v0.8b}, [%[inptr0]] \n"
      "ld1 {v1.8b}, [%[inptr1]] \n"
      "ld1 {v2.8b}, [%[inptr2]] \n"
      "ld1 {v3.8b}, [%[inptr3]] \n"
      "trn1 v4.8b, v0.8b, v1.8b \n"  // v4 = a0b0a2b2a4b4a6b6
      "trn2 v5.8b, v0.8b, v1.8b \n"  // v4 = a1b1a3b3a5b5a7b7
      "trn1 v6.8b, v2.8b, v3.8b \n"  // v4 = c0d0c2d2a4b4a6b6
      "trn2 v7.8b, v2.8b, v3.8b \n"  // v4 = c1d1c3d3a5b5a7b7
      "trn1 v0.4h, v4.4h, v6.4h \n"  // v4 = a0b0c0d0a4b4c4d4
      "trn2 v1.4h, v4.4h, v6.4h \n"  // v4 = a2b2c2d2a6b6c6d6
      "trn1 v2.4h, v5.4h, v7.4h \n"  // v4 = a1b1c1d1a5b5c5d5
      "trn2 v3.4h, v5.4h, v7.4h \n"  // v4 = a3b3c3d3a7b7c7d7
      "trn1 v4.2s, v0.2s, v2.2s \n"  // v4 =a0b0c0d0a1b1c1d1
      "trn2 v5.2s, v0.2s, v2.2s \n"  // v4 =a4b4c4d4a5b5c5d5
      "trn1 v6.2s, v1.2s, v3.2s \n"  // v4 =a2b2c2d2a3b3c3d3
      "trn2 v7.2s, v1.2s, v3.2s \n"  // v4 =a6b6c6d6a7b7c7d7
      "st1 {v4.2s}, [%[outr]], #8\n"
      "st1 {v6.2s}, [%[outr]], #8\n"
      "st1 {v5.2s}, [%[outr]], #8\n"
      "st1 {v7.2s}, [%[outr]], #8\n"
      : [inptr0] "+r"(inptr0),
        [inptr1] "+r"(inptr1),
        [inptr2] "+r"(inptr2),
        [inptr3] "+r"(inptr3),
        [outr] "+r"(out0)
      :
      : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "cc", "memory");
    }
    x += 8;
  }
  if ((x_len - x) >= 4) {
    for (int y = 0; y < y_len; y += 4) {
      inptr = in + k0 * ldin + n0 + x;
      const int8_t* inptr0 = inptr + y * ldin;
      const int8_t* inptr1 = inptr0 + ldin;
      const int8_t* inptr2 = inptr1 + ldin;
      const int8_t* inptr3 = inptr2 + ldin;
      asm volatile(
      "ld1 {v0.8b}, [%[inptr0]] \n"
      "ld1 {v1.8b}, [%[inptr1]] \n"
      "ld1 {v2.8b}, [%[inptr2]] \n"
      "ld1 {v3.8b}, [%[inptr3]] \n"
      "trn1 v4.8b, v0.8b, v1.8b \n"  // v4 = a0b0a2b2a4b4a6b6
      "trn2 v5.8b, v0.8b, v1.8b \n"  // v4 = a1b1a3b3a5b5a7b7
      "trn1 v6.8b, v2.8b, v3.8b \n"  // v4 = c0d0c2d2a4b4a6b6
      "trn2 v7.8b, v2.8b, v3.8b \n"  // v4 = c1d1c3d3a5b5a7b7
      "trn1 v0.4h, v4.4h, v6.4h \n"  // v4 = a0b0c0d0a4b4c4d4
      "trn2 v1.4h, v4.4h, v6.4h \n"  // v4 = a2b2c2d2a6b6c6d6
      "trn1 v2.4h, v5.4h, v7.4h \n"  // v4 = a1b1c1d1a5b5c5d5
      "trn2 v3.4h, v5.4h, v7.4h \n"  // v4 = a3b3c3d3a7b7c7d7
      "trn1 v4.2s, v0.2s, v2.2s \n"  // v4 =a0b0c0d0a1b1c1d1
      "trn1 v6.2s, v1.2s, v3.2s \n"  // v4 =a2b2c2d2a3b3c3d3
      "st1 {v4.2s}, [%[outr]], #8\n"
      "st1 {v6.2s}, [%[outr]], #8\n"
      : [inptr0] "+r"(inptr0),
        [inptr1] "+r"(inptr1),
        [inptr2] "+r"(inptr2),
        [inptr3] "+r"(inptr3),
        [outr] "+r"(out0)
      :
      : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "cc", "memory");
    }
    x += 4;
  }
  // clang-format on
  if (x_len - x) {
    int remain_x = ((x_len - x) + 3) / 4 * 4 + x - x_len;
    for (int y = 0; y < y_len; y += 4) {
      int x_tmp = x;
      for (; x_tmp < x_len; x_tmp++) {
        inptr = in + k0 * ldin + n0 + x_tmp;
        const int8_t* inptr0 = inptr + y * ldin;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        *out0++ = *inptr0++;
        *out0++ = *inptr1++;
        *out0++ = *inptr2++;
        *out0++ = *inptr3++;
      }
      for (int i = 0; i < remain_x; i++) {
        *out0++ = 0;
        *out0++ = 0;
        *out0++ = 0;
        *out0++ = 0;
      }
    }
  }
}

void packb_sdot_int8_n12_n8_n4_trans(int8_t* out,
                                     const int8_t* in,
                                     const int ldin,
                                     const int k0,
                                     const int kmax,
                                     const int n0,
                                     const int nmax) {
  int8_t* outptr = out;
  const int8_t* inptr = in + n0 * ldin + k0;
  int y_len = nmax - n0;
  int x_len = kmax - k0;
  int kup = ROUNDUP(x_len, KBLOCK_INT8);
  int8_t zerobuff[kup];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * kup);
  int stride_y = 48;
  int stride_out = kup;
  int remain = x_len % 8;
  int y = 0;
  for (y = 0; y < y_len - 11; y += 12) {
    const int8_t* inptr_row[12];
    inptr_row[0] = inptr + y * ldin;
    for (int i = 1; i < 12; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }
    if (y + 12 > y_len) {
      for (int i = y + 12 - y_len; i > 0; i--) {
        inptr_row[12 - i] = zerobuff;
      }
    }
    // clang-format off
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr4]]        \n"
        "prfm   pldl1keep, [%[ptr5]]        \n"
        "prfm   pldl1keep, [%[ptr6]]        \n"
        "prfm   pldl1keep, [%[ptr7]]        \n"
        "prfm   pldl1keep, [%[ptr8]]        \n"
        "prfm   pldl1keep, [%[ptr9]]        \n"
        "prfm   pldl1keep, [%[ptr10]]        \n"
        "prfm   pldl1keep, [%[ptr11]]        \n"
        :
        :[ptr0] "r"(inptr_row[0]), [ptr1] "r"(inptr_row[1]),
         [ptr2] "r"(inptr_row[2]), [ptr3] "r"(inptr_row[3]),
         [ptr4] "r"(inptr_row[4]), [ptr5] "r"(inptr_row[5]),
         [ptr6] "r"(inptr_row[6]), [ptr7] "r"(inptr_row[7]),
         [ptr8] "r"(inptr_row[8]), [ptr9] "r"(inptr_row[9]),
         [ptr10] "r"(inptr_row[10]), [ptr11] "r"(inptr_row[11])
        :"memory"
        );
        int right_remain = remain;
        int8_t *outptr_row = outptr + y * stride_out;
        for (int x = 0; x < x_len - 7; x += 8) {
            int8_t *out0 = outptr_row;
            int8_t *out1 = out0 + stride_y;
            asm volatile(
            "ld1  {v0.8b}, [%[inptr0]], #8 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v1.8b}, [%[inptr1]], #8 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v2.8b}, [%[inptr2]], #8 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v3.8b}, [%[inptr3]], #8 \n" // q0=d0d1d2d3A4A5A6A7

            "ld1  {v4.8b}, [%[inptr4]], #8 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v5.8b}, [%[inptr5]], #8 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v6.8b}, [%[inptr6]], #8 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v7.8b}, [%[inptr7]], #8 \n" // q0=d0d1d2d3A4A5A6A7

            "trn1  v8.2s, v0.2s, v1.2s \n"  //v0=a0a1a2a3'b0b1b2b3 -00 01
            "trn2  v12.2s, v0.2s, v1.2s \n"  //v0=a4a5a6a7'b4b5b6b7 - 10 11
            "trn1  v9.2s, v2.2s, v3.2s \n"  //v0=c0c1a2a3'd0b1b2b3 -02 03
            "trn2  v13.2s, v2.2s, v3.2s \n"  //v0=c4a5a6a7'c4b5b6b7 - 12 13

            "ld1  {v0.8b}, [%[inptr8]], #8 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v1.8b}, [%[inptr9]], #8 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v2.8b}, [%[inptr10]], #8 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v3.8b}, [%[inptr11]], #8 \n" // q0=d0d1d2d3A4A5A6A7

            "st1 {v8.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v12.8b}, [%[outptr_row1]], #8 \n"
            "st1 {v9.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v13.8b}, [%[outptr_row1]], #8 \n"

            "trn1  v10.2s, v4.2s, v5.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -04 05
            "trn2  v14.2s, v4.2s, v5.2s \n"  //v0=a2b2a2b2'a6b6a6b6 -14 15
            "trn1  v11.2s, v6.2s, v7.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -06 07
            "trn2  v15.2s, v6.2s, v7.2s \n"  //v0=a2b2a2b2'a6b6a6b6 -16 17

            "trn1  v4.2s, v0.2s, v1.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -08 09
            "trn2  v5.2s, v0.2s, v1.2s \n"  //v0=a2b2a2b2'a6b6a6b6 -18 19
            "trn1  v6.2s, v2.2s, v3.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -010 011
            "trn2  v7.2s, v2.2s, v3.2s \n"  //v0=a2b2a2b2'a6b6a6b6 -110 111

            "st1 {v10.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v14.8b}, [%[outptr_row1]], #8 \n"
            "st1 {v11.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v15.8b}, [%[outptr_row1]], #8 \n"

            "st1 {v4.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v5.8b}, [%[outptr_row1]], #8 \n"
            "st1 {v6.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v7.8b}, [%[outptr_row1]], #8 \n"
            : [inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
              [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
              [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
              [inptr6] "+r"(inptr_row[6]), [inptr7] "+r"(inptr_row[7]),
              [inptr8] "+r"(inptr_row[8]), [inptr9] "+r"(inptr_row[9]),
              [inptr10] "+r"(inptr_row[10]), [inptr11] "+r"(inptr_row[11]),
              [outptr_row0] "+r"(out0), [outptr_row1] "+r"(out1)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
              "v10", "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory"
            );
            outptr_row += 96;
        }
        int8_t *out0 = outptr_row;
        if (right_remain >= 4) {
            asm volatile(
            "mov x1, #4 \n"
            "ld1  {v0.8b}, [%[inptr0]], x1 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v1.8b}, [%[inptr1]], x1 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v2.8b}, [%[inptr2]], x1 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v3.8b}, [%[inptr3]], x1 \n" // q0=d0d1d2d3A4A5A6A7

            "ld1  {v4.8b}, [%[inptr4]], x1 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v5.8b}, [%[inptr5]], x1 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v6.8b}, [%[inptr6]], x1 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v7.8b}, [%[inptr7]], x1 \n" // q0=d0d1d2d3A4A5A6A7

            "trn1  v8.2s, v0.2s, v1.2s \n"  //v0=a0a1a2a3'b0b1b2b3 -00 01
            "trn1  v9.2s, v2.2s, v3.2s \n"  //v0=c0c1a2a3'd0b1b2b3 -02 03

            "ld1  {v12.8b}, [%[inptr8]], x1 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v13.8b}, [%[inptr9]], x1 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v14.8b}, [%[inptr10]], x1 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v15.8b}, [%[inptr11]], x1 \n" // q0=d0d1d2d3A4A5A6A7

            "trn1  v10.2s, v4.2s, v5.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -04 05
            "trn1  v11.2s, v6.2s, v7.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -06 07

            "trn1  v4.2s, v12.2s, v13.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -08 09
            "trn1  v6.2s, v14.2s, v15.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -010 011

            "st1 {v8.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v9.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v10.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v11.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v4.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v6.8b}, [%[outptr_row0]], #8 \n"
            : [inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
            [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
             [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
             [inptr6] "+r"(inptr_row[6]), [inptr7] "+r"(inptr_row[7]),
             [inptr8] "+r"(inptr_row[8]), [inptr9] "+r"(inptr_row[9]),
             [inptr10] "+r"(inptr_row[10]), [inptr11] "+r"(inptr_row[11]), \
              [outptr_row0] "+r"(out0)
            :
            : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
            "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory"
            );
            right_remain -= 4;
        }
    // clang-format on
    if (right_remain > 0) {
      for (int i = 0; i < 12; i++) {
        for (int x = 0; x < right_remain; x++) {
          *out0++ = *inptr_row[i]++;
        }
        for (int x = 0; x < 4 - right_remain; x++) {
          *out0++ = 0;
        }
      }
    }
  }
  int8_t* out0 = outptr + y * stride_out;
  if ((y_len - y) >= 8) {
    for (int x = 0; x < x_len; x += 4) {
      const int8_t* inptr_row[8];
      inptr_row[0] = inptr + y * ldin + x;
      for (int i = 1; i < 8; i++) {
        inptr_row[i] = inptr_row[i - 1] + ldin;
      }
      // clang-format off
      asm volatile(
      "mov x1, #4 \n"
      "ld1  {v0.8b}, [%[inptr0]], x1 \n"  // q0=A0A1A2A3A4A5A6A7
      "ld1  {v1.8b}, [%[inptr1]], x1 \n"  // q0=B0b1b2b3A4A5A6A7
      "ld1  {v2.8b}, [%[inptr2]], x1 \n"  // q0=c0c1c2c3A4A5A6A7
      "ld1  {v3.8b}, [%[inptr3]], x1 \n"  // q0=d0d1d2d3A4A5A6A7
      "ld1  {v4.8b}, [%[inptr4]], x1 \n"  // q0=A0A1A2A3A4A5A6A7
      "ld1  {v5.8b}, [%[inptr5]], x1 \n"  // q0=B0b1b2b3A4A5A6A7
      "ld1  {v6.8b}, [%[inptr6]], x1 \n"  // q0=c0c1c2c3A4A5A6A7
      "ld1  {v7.8b}, [%[inptr7]], x1 \n"  // q0=d0d1d2d3A4A5A6A7
      "trn1  v8.2s, v0.2s, v1.2s \n"      // v0=a0a1a2a3'b0b1b2b3 -00 01
      "trn1  v9.2s, v2.2s, v3.2s \n"      // v0=c0c1a2a3'd0b1b2b3 -02 03
      "trn1  v10.2s, v4.2s, v5.2s \n"     // v0=a0b0a0b0'a4b4a4b4 -04 05
      "trn1  v11.2s, v6.2s, v7.2s \n"     // v0=a0b0a0b0'a4b4a4b4 -06 07
      "st1 {v8.8b}, [%[outptr_row0]], #8 \n"
      "st1 {v9.8b}, [%[outptr_row0]], #8 \n"
      "st1 {v10.8b}, [%[outptr_row0]], #8 \n"
      "st1 {v11.8b}, [%[outptr_row0]], #8 \n"
      : [inptr0] "+r"(inptr_row[0]),
        [inptr1] "+r"(inptr_row[1]),
        [inptr2] "+r"(inptr_row[2]),
        [inptr3] "+r"(inptr_row[3]),
        [inptr4] "+r"(inptr_row[4]),
        [inptr5] "+r"(inptr_row[5]),
        [inptr6] "+r"(inptr_row[6]),
        [inptr7] "+r"(inptr_row[7]),
        [outptr_row0] "+r"(out0)
      :
      : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
      "v9", "v10", "v11", "memory");
      // clang-format on
    }
    y += 8;
  }
  if ((y_len - y) >= 4) {
    for (int x = 0; x < x_len; x += 4) {
      const int8_t* inptr_row[4];
      inptr_row[0] = inptr + y * ldin + x;
      for (int i = 1; i < 4; i++) {
        inptr_row[i] = inptr_row[i - 1] + ldin;
      }
      // clang-format off
      asm volatile(
      "mov x1, #4 \n"
      "ld1  {v0.8b}, [%[inptr0]], x1 \n"  // q0=A0A1A2A3A4A5A6A7
      "ld1  {v1.8b}, [%[inptr1]], x1 \n"  // q0=B0b1b2b3A4A5A6A7
      "ld1  {v2.8b}, [%[inptr2]], x1 \n"  // q0=c0c1c2c3A4A5A6A7
      "ld1  {v3.8b}, [%[inptr3]], x1 \n"  // q0=d0d1d2d3A4A5A6A7
      "trn1  v4.2s, v0.2s, v1.2s \n"      // v0=a0a1a2a3'b0b1b2b3 -00 01
      "trn1  v5.2s, v2.2s, v3.2s \n"      // v0=c0c1a2a3'd0b1b2b3 -02 03
      "st1 {v4.8b}, [%[outptr_row0]], #8 \n"
      "st1 {v5.8b}, [%[outptr_row0]], #8 \n"
      : [inptr0] "+r"(inptr_row[0]),
        [inptr1] "+r"(inptr_row[1]),
        [inptr2] "+r"(inptr_row[2]),
        [inptr3] "+r"(inptr_row[3]),
        [outptr_row0] "+r"(out0)
      :
      : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "memory");
      // clang-format on
    }
    y += 4;
  }
  if (y_len - y) {
    for (int x = 0; x < x_len; x += 4) {
      const int8_t* inptr_row[4];
      inptr_row[0] = inptr + y * ldin + x;
      for (int i = 1; i < 4; i++) {
        inptr_row[i] = inptr_row[i - 1] + ldin;
      }
      if (y + 4 > y_len) {
        for (int i = y + 4 - y_len; i > 0; i--) {
          inptr_row[4 - i] = zerobuff;
        }
      }
      // clang-format off
      asm volatile(
      "mov x1, #4 \n"
      "ld1  {v0.8b}, [%[inptr0]], x1 \n"  // q0=A0A1A2A3A4A5A6A7
      "ld1  {v1.8b}, [%[inptr1]], x1 \n"  // q0=B0b1b2b3A4A5A6A7
      "ld1  {v2.8b}, [%[inptr2]], x1 \n"  // q0=c0c1c2c3A4A5A6A7
      "ld1  {v3.8b}, [%[inptr3]], x1 \n"  // q0=d0d1d2d3A4A5A6A7
      "trn1  v4.2s, v0.2s, v1.2s \n"      // v0=a0a1a2a3'b0b1b2b3 -00 01
      "trn1  v5.2s, v2.2s, v3.2s \n"      // v0=c0c1a2a3'd0b1b2b3 -02 03
      "st1 {v4.8b}, [%[outptr_row0]], #8 \n"
      "st1 {v5.8b}, [%[outptr_row0]], #8 \n"
      : [inptr0] "+r"(inptr_row[0]),
        [inptr1] "+r"(inptr_row[1]),
        [inptr2] "+r"(inptr_row[2]),
        [inptr3] "+r"(inptr_row[3]),
        [outptr_row0] "+r"(out0)
      :
      : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "memory");
      // clang-format on
    }
  }
}

void packb_sdot_int8(int8_t* out,
                     const int8_t* in,
                     const int ldin,
                     const int k0,
                     const int kmax,
                     const int n0,
                     const int nmax) {
  int y_len = kmax - k0;
  int x_len = nmax - n0;
  int kup = ROUNDUP(y_len, KBLOCK_INT8);  //  4k
  int8_t zerobuff[x_len];                 // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * x_len);
  int8_t* outptr = out;
  const int8_t* inptr = in + k0 * ldin + n0;

  int stride_out = 12 * kup;
  int remain = x_len % 12;

  // data B is not transposed, transpose B to k * 12
  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, 0, 4) {
    // cope with row index exceed real size, set to zero
    const int8_t* inptr0 = inptr + y * ldin;
    const int8_t* inptr1 = inptr0 + ldin;
    const int8_t* inptr2 = inptr1 + ldin;
    const int8_t* inptr3 = inptr2 + ldin;
    if (y + 4 > y_len) {
      switch (y + 4 - y_len) {
        case 3:
          inptr1 = zerobuff;
        case 2:
          inptr2 = zerobuff;
        case 1:
          inptr3 = zerobuff;
        default:
          break;
      }
    }
    // clang-format off
        asm volatile(
        "prfm   pldl1keep, [%[inptr0]]                \n"
        "prfm   pldl1keep, [%[inptr0], #64]        \n"
        "prfm   pldl1keep, [%[inptr1]]        \n"
        "prfm   pldl1keep, [%[inptr1], #64]        \n"
        "prfm   pldl1keep, [%[inptr2]]        \n"
        "prfm   pldl1keep, [%[inptr2], #64]        \n"
        "prfm   pldl1keep, [%[inptr3]]        \n"
        "prfm   pldl1keep, [%[inptr3], #64]        \n"
        :
        :[inptr0] "r"(inptr0), [inptr1] "r"(inptr1),
          [inptr2] "r"(inptr2), [inptr3] "r"(inptr3)
        :"memory"
        );
        int8_t* outptr_row = outptr + y * 12;
        int x = 0;
        for (; x < x_len - 11; x += 12) {
            int8_t *out0 = outptr_row;
            asm volatile (
            "mov x1, #4 \n"
            "ld1 {v0.8b}, [%[inptr0]], #8 \n" // v0 = a0a1a2a3a4a5a6a7
            "ld1 {v1.8b}, [%[inptr1]], #8 \n" // v0 = b0b1b2b3b4b5b6b7
            "ld1 {v2.8b}, [%[inptr2]], #8 \n" // v0 = c0c1c2c3c4c5c6c7
            "ld1 {v3.8b}, [%[inptr3]], #8 \n" // v0 = d0d1d2d3d4d5d6d7

            "ld1 {v8.8b}, [%[inptr0]]  \n" // v0 = a8a9a10a11
            "ld1 {v9.8b}, [%[inptr1]]  \n" // v0 = b8b9b10b11
            "ld1 {v10.8b}, [%[inptr2]]  \n" // v0 = c8c9c10c11
            "ld1 {v11.8b}, [%[inptr3]]  \n" // v0 = d8d9d10d11

            "trn1 v4.8b, v0.8b, v1.8b \n" // v4 = a0b0a2b2a4b4a6b6
            "trn2 v5.8b, v0.8b, v1.8b \n" // v4 = a1b1a3b3a5b5a7b7
            "trn1 v6.8b, v2.8b, v3.8b \n" // v4 = c0d0c2d2a4b4a6b6
            "trn2 v7.8b, v2.8b, v3.8b \n" // v4 = c1d1c3d3a5b5a7b7

            "trn1 v12.8b, v8.8b, v9.8b \n" // v4 = a8b8a10b10a4b4a6b6
            "trn2 v13.8b, v8.8b, v9.8b \n" // v4 = a9b9a11b11a5b5a7b7
            "trn1 v14.8b, v10.8b, v11.8b \n" // v4 = c8d8c10d10a4b4a6b6
            "trn2 v15.8b, v10.8b, v11.8b \n" // v4 = c9d9c11d11a5b5a7b7

            "trn1 v0.4h, v4.4h, v6.4h \n" // v4 = a0b0c0d0a4b4c4d4
            "trn2 v1.4h, v4.4h, v6.4h \n" // v4 = a2b2c2d2a6b6c6d6
            "trn1 v2.4h, v5.4h, v7.4h \n" // v4 = a1b1c1d1a5b5c5d5
            "trn2 v3.4h, v5.4h, v7.4h \n" // v4 = a3b3c3d3a7b7c7d7

            "trn1 v8.4h, v12.4h, v14.4h \n" // v4 = a8b8c8d8
            "trn2 v9.4h, v12.4h, v14.4h \n" // v4 = a10b10c10d10
            "trn1 v10.4h, v13.4h, v15.4h \n" // v4 = a9b9c9d9
            "trn2 v11.4h, v13.4h, v15.4h \n" // v4 = a11b11c11d11

            "trn1 v4.2s, v0.2s, v2.2s \n" //v4 =a0b0c0d0a1b1c1d1
            "trn2 v5.2s, v0.2s, v2.2s \n" //v4 =a4b4c4d4a5b5c5d5
            "trn1 v6.2s, v1.2s, v3.2s \n" //v4 =a2b2c2d2a3b3c3d3
            "trn2 v7.2s, v1.2s, v3.2s \n" //v4 =a6b6c6d6a7b7c7d7

            "trn1 v0.2s, v8.2s, v10.2s \n" //v4 =a8b8c8d8a9b9c9d9
            "trn1 v1.2s, v9.2s, v11.2s \n" //v4 =a10b10c10d10a11b11c11d11

            "st1 {v4.2s}, [%[outr]], #8\n"
            "st1 {v6.2s}, [%[outr]], #8\n"
            "add %[inptr0], %[inptr0], #4\n"
            "add %[inptr1], %[inptr1], #4\n"
            "st1 {v5.2s}, [%[outr]], #8\n"
            "st1 {v7.2s}, [%[outr]], #8\n"
            "add %[inptr2], %[inptr2], #4\n"
            "add %[inptr3], %[inptr3], #4\n"
            "st1 {v0.2s}, [%[outr]], #8\n"
            "st1 {v1.2s}, [%[outr]], #8\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outr] "+r"(out0)
            :
            : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "cc", "memory"
            );
            outptr_row += stride_out;
        }
    // clang-format on
    int8_t* out0 = outptr_row;  //  outptr + stride_out + y * remain;
    for (; x < x_len; x++) {
      *out0++ = *inptr0++;
      *out0++ = *inptr1++;
      *out0++ = *inptr2++;
      *out0++ = *inptr3++;
    }
    for (int i = 0; i < 12 - remain; i++) {
      *out0++ = 0;
      *out0++ = 0;
      *out0++ = 0;
      *out0++ = 0;
    }
  }
  LITE_PARALLEL_COMMON_END();
}

void packb_sdot_trans_int8(int8_t* out,
                           const int8_t* in,
                           const int ldin,
                           const int k0,
                           const int kmax,
                           const int n0,
                           const int nmax) {
  int8_t* outptr = out;
  const int8_t* inptr = in + n0 * ldin + k0;
  int y_len = nmax - n0;
  int x_len = kmax - k0;

  int kup = ROUNDUP(x_len, KBLOCK_INT8);  //  4

  int8_t zerobuff[kup];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * kup);

  int stride_y = 48;
  int stride_out = kup;

  int remain = x_len % 8;
  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, 0, 12) {
    const int8_t* inptr_row[12];
    inptr_row[0] = inptr + y * ldin;
    for (int i = 1; i < 12; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }
    if (y + 12 > y_len) {
      for (int i = y + 12 - y_len; i > 0; i--) {
        inptr_row[12 - i] = zerobuff;
      }
    }
    // clang-format off
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr4]]        \n"
        "prfm   pldl1keep, [%[ptr5]]        \n"
        "prfm   pldl1keep, [%[ptr6]]        \n"
        "prfm   pldl1keep, [%[ptr7]]        \n"
        "prfm   pldl1keep, [%[ptr8]]        \n"
        "prfm   pldl1keep, [%[ptr9]]        \n"
        "prfm   pldl1keep, [%[ptr10]]        \n"
        "prfm   pldl1keep, [%[ptr11]]        \n"
        :
        :[ptr0] "r"(inptr_row[0]), [ptr1] "r"(inptr_row[1]),
         [ptr2] "r"(inptr_row[2]), [ptr3] "r"(inptr_row[3]),
         [ptr4] "r"(inptr_row[4]), [ptr5] "r"(inptr_row[5]),
         [ptr6] "r"(inptr_row[6]), [ptr7] "r"(inptr_row[7]),
         [ptr8] "r"(inptr_row[8]), [ptr9] "r"(inptr_row[9]),
         [ptr10] "r"(inptr_row[10]), [ptr11] "r"(inptr_row[11])
        :"memory"
        );
        int right_remain = remain;
        int8_t *outptr_row = outptr + y * stride_out;
        for (int x = 0; x < x_len - 7; x += 8) {
            int8_t *out0 = outptr_row;
            int8_t *out1 = out0 + stride_y;
            asm volatile(
            "ld1  {v0.8b}, [%[inptr0]], #8 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v1.8b}, [%[inptr1]], #8 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v2.8b}, [%[inptr2]], #8 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v3.8b}, [%[inptr3]], #8 \n" // q0=d0d1d2d3A4A5A6A7

            "ld1  {v4.8b}, [%[inptr4]], #8 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v5.8b}, [%[inptr5]], #8 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v6.8b}, [%[inptr6]], #8 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v7.8b}, [%[inptr7]], #8 \n" // q0=d0d1d2d3A4A5A6A7

            "trn1  v8.2s, v0.2s, v1.2s \n"  //v0=a0a1a2a3'b0b1b2b3 -00 01
            "trn2  v12.2s, v0.2s, v1.2s \n"  //v0=a4a5a6a7'b4b5b6b7 - 10 11
            "trn1  v9.2s, v2.2s, v3.2s \n"  //v0=c0c1a2a3'd0b1b2b3 -02 03
            "trn2  v13.2s, v2.2s, v3.2s \n"  //v0=c4a5a6a7'c4b5b6b7 - 12 13

            "ld1  {v0.8b}, [%[inptr8]], #8 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v1.8b}, [%[inptr9]], #8 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v2.8b}, [%[inptr10]], #8 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v3.8b}, [%[inptr11]], #8 \n" // q0=d0d1d2d3A4A5A6A7

            "st1 {v8.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v12.8b}, [%[outptr_row1]], #8 \n"
            "st1 {v9.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v13.8b}, [%[outptr_row1]], #8 \n"

            "trn1  v10.2s, v4.2s, v5.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -04 05
            "trn2  v14.2s, v4.2s, v5.2s \n"  //v0=a2b2a2b2'a6b6a6b6 -14 15
            "trn1  v11.2s, v6.2s, v7.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -06 07
            "trn2  v15.2s, v6.2s, v7.2s \n"  //v0=a2b2a2b2'a6b6a6b6 -16 17

            "trn1  v4.2s, v0.2s, v1.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -08 09
            "trn2  v5.2s, v0.2s, v1.2s \n"  //v0=a2b2a2b2'a6b6a6b6 -18 19
            "trn1  v6.2s, v2.2s, v3.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -010 011
            "trn2  v7.2s, v2.2s, v3.2s \n"  //v0=a2b2a2b2'a6b6a6b6 -110 111

            "st1 {v10.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v14.8b}, [%[outptr_row1]], #8 \n"
            "st1 {v11.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v15.8b}, [%[outptr_row1]], #8 \n"

            "st1 {v4.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v5.8b}, [%[outptr_row1]], #8 \n"
            "st1 {v6.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v7.8b}, [%[outptr_row1]], #8 \n"
            : [inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
              [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
              [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
              [inptr6] "+r"(inptr_row[6]), [inptr7] "+r"(inptr_row[7]),
              [inptr8] "+r"(inptr_row[8]), [inptr9] "+r"(inptr_row[9]),
              [inptr10] "+r"(inptr_row[10]), [inptr11] "+r"(inptr_row[11]),
              [outptr_row0] "+r"(out0), [outptr_row1] "+r"(out1)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
              "v10", "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory"
            );
            outptr_row += 96;
        }
        int8_t *out0 = outptr_row;
        if (right_remain >= 4) {
            asm volatile(
            "mov x1, #4 \n"
            "ld1  {v0.8b}, [%[inptr0]], x1 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v1.8b}, [%[inptr1]], x1 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v2.8b}, [%[inptr2]], x1 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v3.8b}, [%[inptr3]], x1 \n" // q0=d0d1d2d3A4A5A6A7

            "ld1  {v4.8b}, [%[inptr4]], x1 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v5.8b}, [%[inptr5]], x1 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v6.8b}, [%[inptr6]], x1 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v7.8b}, [%[inptr7]], x1 \n" // q0=d0d1d2d3A4A5A6A7

            "trn1  v8.2s, v0.2s, v1.2s \n"  //v0=a0a1a2a3'b0b1b2b3 -00 01
            "trn1  v9.2s, v2.2s, v3.2s \n"  //v0=c0c1a2a3'd0b1b2b3 -02 03

            "ld1  {v12.8b}, [%[inptr8]], x1 \n" // q0=A0A1A2A3A4A5A6A7
            "ld1  {v13.8b}, [%[inptr9]], x1 \n" // q0=B0b1b2b3A4A5A6A7
            "ld1  {v14.8b}, [%[inptr10]], x1 \n" // q0=c0c1c2c3A4A5A6A7
            "ld1  {v15.8b}, [%[inptr11]], x1 \n" // q0=d0d1d2d3A4A5A6A7

            "trn1  v10.2s, v4.2s, v5.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -04 05
            "trn1  v11.2s, v6.2s, v7.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -06 07

            "trn1  v4.2s, v12.2s, v13.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -08 09
            "trn1  v6.2s, v14.2s, v15.2s \n"  //v0=a0b0a0b0'a4b4a4b4 -010 011

            "st1 {v8.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v9.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v10.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v11.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v4.8b}, [%[outptr_row0]], #8 \n"
            "st1 {v6.8b}, [%[outptr_row0]], #8 \n"
            : [inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
            [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
             [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
             [inptr6] "+r"(inptr_row[6]), [inptr7] "+r"(inptr_row[7]),
             [inptr8] "+r"(inptr_row[8]), [inptr9] "+r"(inptr_row[9]),
             [inptr10] "+r"(inptr_row[10]), [inptr11] "+r"(inptr_row[11]), \
              [outptr_row0] "+r"(out0)
            :
            : "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
            "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory"
            );
            right_remain -= 4;
        }
    // clang-format on
    if (right_remain > 0) {
      for (int i = 0; i < 12; i++) {
        for (int x = 0; x < right_remain; x++) {
          *out0++ = *inptr_row[i]++;
        }
        for (int x = 0; x < 4 - right_remain; x++) {
          *out0++ = 0;
        }
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
}
#else  // armv7

void prepackA_m6k4_int8(int8_t* out,
                        const int8_t* in,
                        const int ldin,
                        const int m0,
                        const int mmax,
                        const int k0,
                        const int kmax) {
  int x_len = (kmax - k0);
  int8_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * x_len);
  int8_t* dout = out;
  const int8_t* inptr = in;
  int kup = ROUNDUP(x_len, KBLOCK_INT8);
  int stride = kup * 6;
  int remain = x_len % 4;
  LITE_PARALLEL_COMMON_BEGIN(y, tid, mmax, m0, 6) {
    int8_t* outptr = dout + stride * (y - m0) / 6;
    const int8_t* inptr_row[6];
    inptr_row[0] = inptr + y * ldin + k0;
    for (int i = 1; i < 6; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }
    //! cope with row index exceed real size, set to zero buffer
    if ((y + 5) >= mmax) {
      switch ((y + 5) - mmax) {
        case 4:
          inptr_row[1] = zerobuff;
        case 3:
          inptr_row[2] = zerobuff;
        case 2:
          inptr_row[3] = zerobuff;
        case 1:
          inptr_row[4] = zerobuff;
        case 0:
          inptr_row[5] = zerobuff;
        default:
          break;
      }
    }
    // clang-format off
        int x = x_len;
        for (; x > 15; x -= 16) {
            asm volatile(
            "vld1.s8 {d0-d1}, [%[inptr0]]! \n" // q0=a0~a3a4~a7 a8~a11a12~a15
            "vld1.s8 {d2-d3}, [%[inptr1]]! \n" // q1=b0~b3b4~b7 b8~b11b12~b15
            "vld1.s8 {d4-d5}, [%[inptr2]]! \n" // q2=c0~c3c4~c7 c8~c11c12~c15
            "vld1.s8 {d6-d7}, [%[inptr3]]! \n" // q3=d0~d3d4~d7 d8~d11d12~d15

            "vld1.s8 {d8-d9}, [%[inptr4]]! \n" // q4=e0~e3e4~e7 e8~e11e12~e15
            "vld1.s8 {d10-d11}, [%[inptr5]]! \n" // q5=f0~f3f4~f7 f8~f11f12~f15

            "vtrn.32 q0, q1 \n" // q0=a0~a3b0~b3 a8~a11b8~b11 q1=a4~a7b4~b7 a12~a15b12~b15
            "vtrn.32 q2, q3 \n" // q2=c0~c3d0~d3 c8~c11d8~d11 q3=c4~c7d4~d7 c12~c15d12~d15
            "vtrn.32 q4, q5 \n" // q4=e0~e3f0~f3 e8~e11f8~f11 q5=e4~e7f4~f7 e8~e11f12~f15

            "vswp     d1, d4   \n"  // q0=a0~a3b0~b3 c0~c3d0~d3 q2=a8~a11b8~b11 c8~c11d8~d11
            "vswp     d2, d9   \n"  // q1=e8~e11f8~f11 a12~a15b12~b15 q4=e0~e3f0~f3 a4~a7b4~b7
            "vswp     d7, d10   \n" // q3=c4~c7d4~d7 e4~e7f4~f7 q5=c12~c15d12~d15 e8~e11f12~f15

            "vst1.32 {d0-d1}, [%[outptr]]!\n"
            "vst1.32 {d8-d9}, [%[outptr]]!\n"
            "vst1.32 {d6-d7}, [%[outptr]]!\n"
            "vst1.32 {d4-d5}, [%[outptr]]!\n"
            "vst1.32 {d2-d3}, [%[outptr]]!\n"
            "vst1.32 {d10-d11}, [%[outptr]]!\n"

            :[inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
            [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
            [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
            [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5",
             "cc", "memory"
            );
        }
        for (; x > 7; x -= 8) {
            asm volatile(
            "vld1.s8 {d0}, [%[inptr0]]! \n" // d0=a0a1a2a3a4a5a6a7
            "vld1.s8 {d2}, [%[inptr1]]! \n" // d2=c0c1c2c3c4c5c6c7
            "vld1.s8 {d1}, [%[inptr2]]! \n" // d1=b0b1b2b3b4b5b6b7
            "vld1.s8 {d3}, [%[inptr3]]! \n" // d3=d0d1d2d3d4d5d6d7

            "vld1.s8 {d4}, [%[inptr4]]! \n" // d4=e0e1e2e3e4e5e6e7
            "vld1.s8 {d5}, [%[inptr5]]! \n" // d5=f0f1f2f3f4f5f6f7

            "vtrn.32 q0, q1 \n" // q0=a0~a3b0~b3 c0~c3d0~d3 q1=a4~a7b4~b7 c4~c7d4~d7
            "vtrn.32 d4, d5 \n" // q2=e0~e3f0~f3 e4~e7f8~f11 

            "vst1.32 {d0-d1}, [%[outptr]]!\n"
            "vst1.32 {d4}, [%[outptr]]!\n"
            "vst1.32 {d2-d3}, [%[outptr]]!\n"
            "vst1.32 {d5}, [%[outptr]]!\n"

            :[inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
            [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
            [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
            [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q4",
             "cc", "memory"
            );
        }
            int8_t* outptr_temp =  outptr;
        if (x > 3) {
            asm volatile(
            "mov r1, #4 \n"
            "vld1.s8 {d0}, [%[inptr0]], r1 \n" // d0=a0a1a2a3a4a5a6a7
            "vld1.s8 {d2}, [%[inptr1]], r1 \n" // d2=c0c1c2c3c4c5c6c7
            "vld1.s8 {d1}, [%[inptr2]], r1 \n" // d1=b0b1b2b3b4b5b6b7
            "vld1.s8 {d3}, [%[inptr3]], r1 \n" // d3=d0d1d2d3d4d5d6d7


            "vld1.s8 {d4}, [%[inptr4]], r1 \n" // d4=e0e1e2e3e4e5e6e7
            "vld1.s8 {d5}, [%[inptr5]], r1 \n" // d5=f0f1f2f3f4f5f6f7

            "vtrn.32 q0, q1 \n" // q0=a0~a3b0~b3 c0~c3d0~d3 q1=a4~a7b4~b7 c4~c7d4~d7
            "vtrn.32 d4, d5 \n" // q2=e0~e3f0~f3 


            "vst1.32 {d0-d1}, [%[outptr]]!\n"
            "vst1.32 {d4}, [%[outptr]]!\n"

            :[inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
            [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
            [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
            [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q4", "r1", "cc", "memory"
            );
            x -= 4;
        }

    // clang-format on

    if (x > 0) {
      for (int i = 0; i < 6; i++) {
        for (int j = x; j > 0; j--) {
          *outptr++ = *inptr_row[i]++;
        }
        for (int j = 0; j < 4 - remain; j++) {
          *outptr++ = 0;
        }
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
}
void prepackA_m6k4_trans_int8(int8_t* out,
                              const int8_t* in,
                              const int ldin,
                              const int m0,
                              const int mmax,
                              const int k0,
                              const int kmax) {
  int8_t* outptr = out;
  const int8_t* inptr = in + k0 * ldin + m0;
  int x_len = mmax - m0;
  int y_len = kmax - k0;
  int right_remain = x_len % 6;
  int kup = ROUNDUP(y_len, KBLOCK_INT8);
  int stride_out = 6 * kup;
  int8_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * x_len);

  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, 0, 4) {
    const int8_t* inptr0 = inptr + y * ldin;
    const int8_t* inptr1 = inptr0 + ldin;
    const int8_t* inptr2 = inptr1 + ldin;
    const int8_t* inptr3 = inptr2 + ldin;

    if (y + 4 > y_len) {
      switch (y + 4 - y_len) {
        case 3:
          inptr1 = zerobuff;
        case 2:
          inptr2 = zerobuff;
        case 1:
          inptr3 = zerobuff;

        default:
          break;
      }
    }

    // clang-format off
    asm volatile(
    "pld    [%[inptr0]]       \n"
    "pld   [%[inptr1]]        \n"
    "pld   [%[inptr2]]        \n"
    "pld   [%[inptr3]]        \n"
    :
    :[inptr0] "r"(inptr0), [inptr1] "r"(inptr1),
      [inptr2] "r"(inptr2), [inptr3] "r"(inptr3)
    :"memory"
    );
    int8_t *outptr_row = outptr + y * 6;
    int x = x_len;
    for (; x > 5; x -= 6) {
        int8_t *out0 = outptr_row;
        asm volatile(
        "mov r1, #6 \n"
        "vld1.s8 {d0}, [%[inptr0]], r1\n" // d0=a0a1a2a3a4a5a6a7
        "vld1.s8 {d2}, [%[inptr1]], r1\n" // d2=b0b1b2b3b4b5b6b7
        "vld1.s8 {d1}, [%[inptr2]], r1\n" // d1=c0c1c2c3c4c5c6c7
        "vld1.s8 {d3}, [%[inptr3]], r1\n" // d3=d0d1d2d3d4d5d6d7

        "vtrn.8 q0, q1 \n" // q0=a0b0a2b2a4b4a6d6 c0d0c2d2c4d4c6d6
                           // q1=a1b1a3b3a5c5a7b7 c1d1c3d3c5d5c7d7
        
        "vtrn.16 d0, d1 \n" // q0=a0b0c0d0a4b4c4d4 a2b2c2d2a6b6c6d6
        "vtrn.16 d2, d3 \n" // q1=a1b1c1d1a5b5c5d5 a3b3c3d3a7b7c7d7
        
        "vtrn.32 q0, q1 \n" // q0=a0b0c0d0a1b1c1d1 a2b2c2d2a3b3c3d3
                            // q1=a4b4c4d4a5b5c5d5 a6b6c6d6a7b7c7d7
        "vst1.32 {d0-d1}, [%[outptr]]!\n"
        "vst1.32 {d2}, [%[outptr]]!\n"

        :[inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
        [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
        [outptr] "+r"(out0)
        :
        : "q0", "q1", "q2", "r1", "cc", "memory"
        );
        outptr_row += stride_out;
    }
    // clang-format on
    if (right_remain > 0) {
      int8_t* out0 = outptr_row;
      for (; x > 0; x--) {
        *out0++ = *inptr0++;
        *out0++ = *inptr1++;
        *out0++ = *inptr2++;
        *out0++ = *inptr3++;
      }
      for (int i = 0; i < 6 - right_remain; i++) {
        *out0++ = 0;
        *out0++ = 0;
        *out0++ = 0;
        *out0++ = 0;
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
}
void packb_dot_int8(int8_t* out,
                    const int8_t* in,
                    const int ldin,
                    const int k0,
                    const int kmax,
                    const int n0,
                    const int nmax) {
  int y_len = kmax - k0;
  int x_len = nmax - n0;
  int kup = ROUNDUP(y_len, KBLOCK_INT8);  //  4k
  int8_t zerobuff[x_len];                 // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * x_len);
  int8_t* outptr = out;
  const int8_t* inptr = in + k0 * ldin + n0;

  int stride_out = 8 * kup;
  int remain = x_len % 8;

  // data B is not transposed, transpose B to k * 8
  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, 0, 4) {
    // cope with row index exceed real size, set to zero
    const int8_t* inptr0 = inptr + y * ldin;
    const int8_t* inptr1 = inptr0 + ldin;
    const int8_t* inptr2 = inptr1 + ldin;
    const int8_t* inptr3 = inptr2 + ldin;
    if (y + 4 > y_len) {
      switch (y + 4 - y_len) {
        case 3:
          inptr1 = zerobuff;
        case 2:
          inptr2 = zerobuff;
        case 1:
          inptr3 = zerobuff;
        default:
          break;
      }
    }
    // clang-format off
        asm volatile(
        "pld   [%[inptr0]]        \n"
        "pld   [%[inptr1]]        \n"
        "pld   [%[inptr2]]        \n"
        "pld   [%[inptr3]]        \n"
        :
        :[inptr0] "r"(inptr0), [inptr1] "r"(inptr1),
          [inptr2] "r"(inptr2), [inptr3] "r"(inptr3)
        :"memory"
        );
        int8_t* outptr_row = outptr + y * 8;
        int x = 0;
        for (; x < x_len - 7; x += 8) {
            int8_t *out0 = outptr_row;
            asm volatile (
            "vld1.s8 {d0}, [%[inptr0]]! \n" // d0=a0a1a2a3a4a5a6a7
            "vld1.s8 {d1}, [%[inptr1]]! \n" // d1=c0c1c2c3c4c5c6c7
            "vld1.s8 {d2}, [%[inptr2]]! \n" // d2=b0b1b2b3b4b5b6b7
            "vld1.s8 {d3}, [%[inptr3]]! \n" // d3=d0d1d2d3d4d5d6d7

            "vtrn.8 d0, d1 \n" // d0=a0b0a2b2a4b4a6b6 d1=a1b1a3b3a5b5a7b7
            "vtrn.8 d2, d3 \n" // d2=c0d0c2d2c4d4c6d6 d3=c1d1c3d3c5d5c7d7

            "vtrn.16 d0, d2 \n" // d0=a0b0c0d0a4b4c4d4 d2=a2b2c2d2a6b6c6d6
            "vtrn.16 d1, d3 \n" // d1=a1bac1d1a5b5c5d5 d3=a3b3c3d3a7b7c7d7

            "vtrn.32 d0, d1 \n" // d0=a0b0c0d0a1bac1d1 d1=a4b4c4d4a5b5c5d5
            "vtrn.32 d2, d3 \n" // d2=a2b2c2d2a3b3c3d3 d3=a6b6c6d6a7b7c7d7

            "vst1.32 {d0}, [%[outptr]]!\n"
            "vst1.32 {d2}, [%[outptr]]!\n"
            "vst1.32 {d1}, [%[outptr]]!\n"
            "vst1.32 {d3}, [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(out0)
            :
            : "q0", "q1", "q2", "q3", "cc", "memory"
            );
            outptr_row += stride_out;
        }

    // clang-format on
    int8_t* out0 = outptr_row;
    for (; x < x_len; x++) {
      *out0++ = *inptr0++;
      *out0++ = *inptr1++;
      *out0++ = *inptr2++;
      *out0++ = *inptr3++;
    }
    for (int i = 0; i < 8 - remain; i++) {
      *out0++ = 0;
      *out0++ = 0;
      *out0++ = 0;
      *out0++ = 0;
    }
  }
  LITE_PARALLEL_COMMON_END();
}

void packb_dot_trans_int8(int8_t* out,
                          const int8_t* in,
                          const int ldin,
                          const int k0,
                          const int kmax,
                          const int n0,
                          const int nmax) {
  int8_t* outptr = out;
  const int8_t* inptr = in + n0 * ldin + k0;
  int y_len = nmax - n0;
  int x_len = kmax - k0;

  int kup = ROUNDUP(x_len, KBLOCK_INT8);  //  4

  int8_t zerobuff[kup];  // NOLINT
  memset(zerobuff, 0, sizeof(int8_t) * kup);

  int stride_y = 4 * 8;
  int stride_out = kup;

  int remain = x_len % 8;

  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, 0, 8) {
    const int8_t* inptr_row[8];
    inptr_row[0] = inptr + y * ldin;
    for (int i = 1; i < 8; i++) {
      inptr_row[i] = inptr_row[i - 1] + ldin;
    }
    if (y + 8 > y_len) {
      for (int i = y + 8 - y_len; i > 0; i--) {
        inptr_row[8 - i] = zerobuff;
      }
    }
    // clang-format off
        asm volatile(
        "pld [%[ptr0]]        \n"
        "pld [%[ptr1]]        \n"
        "pld [%[ptr2]]        \n"
        "pld [%[ptr3]]        \n"
        "pld [%[ptr4]]        \n"
        "pld [%[ptr5]]        \n"
        "pld [%[ptr6]]        \n"
        "pld [%[ptr7]]        \n"
        :
        :[ptr0] "r"(inptr_row[0]), [ptr1] "r"(inptr_row[1]),
         [ptr2] "r"(inptr_row[2]), [ptr3] "r"(inptr_row[3]),
         [ptr4] "r"(inptr_row[4]), [ptr5] "r"(inptr_row[5]),
         [ptr6] "r"(inptr_row[6]), [ptr7] "r"(inptr_row[7])
        :"memory"
        );

        int right_remain = remain;
        int8_t *outptr_row = outptr + y * stride_out;
        for (int x = 0; x < x_len - 7; x += 8) {
            int8_t *out0 = outptr_row;
            int8_t *out1 = out0 + stride_y;
            asm volatile(
            "vld1.s8 {d0}, [%[inptr0]]! \n" // d0=a0a1a2a3a4a5a6a7
            "vld1.s8 {d4}, [%[inptr1]]! \n" // d4=c0c1c2c3c4c5c6c7
            "vld1.s8 {d1}, [%[inptr2]]! \n" // d1=b0b1b2b3b4b5b6b7
            "vld1.s8 {d5}, [%[inptr3]]! \n" // d5=d0d1d2d3d4d5d6d7

            "vld1.s8 {d2}, [%[inptr4]]! \n" // d2=e0e1e2e3e4e5e6e7
            "vld1.s8 {d6}, [%[inptr5]]! \n" // d6=f0f1f2f3f4f5f6f7
            "vld1.s8 {d3}, [%[inptr6]]! \n" // d3=g0g1g2g3g4g5g6g7
            "vld1.s8 {d7}, [%[inptr7]]! \n" // d7=h0h1h2h3h4h5h6h7

            "vtrn.32  q0, q2 \n"  //q0=a0~a3b0~b3c0~c3d0~d3 q2=a4~a7b4~b7c4~c7d4~d7
            "vtrn.32  q1, q3 \n"  //q1=e0~e3f0~f3g0~g3h0~h3 q3=e4~e7f4~f7g4~g7h4~h7

            "vst1.32 {d0,d1,d2,d3}, [%[outptr_row0]]! \n"
            "vst1.32 {d4,d5,d6,d7}, [%[outptr_row1]]! \n"

            : [inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
              [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
              [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
              [inptr6] "+r"(inptr_row[6]), [inptr7] "+r"(inptr_row[7]),
              [outptr_row0] "+r"(out0), [outptr_row1] "+r"(out1)
            :
            : "q0", "q1", "q2", "q3", "cc", "memory"
            );
            outptr_row += 8*8;
        }
        int8_t *out0 = outptr_row;
        if (right_remain >= 4) {
            asm volatile(
            "mov r1, #4 \n"
            "vld1.s8 {d0}, [%[inptr0]], r1 \n" // d0=a0a1a2a3a4a5a6a7
            "vld1.s8 {d4}, [%[inptr1]], r1 \n" // d4=c0c1c2c3c4c5c6c7
            "vld1.s8 {d1}, [%[inptr2]], r1 \n" // d1=b0b1b2b3b4b5b6b7
            "vld1.s8 {d5}, [%[inptr3]], r1 \n" // d5=d0d1d2d3d4d5d6d7

            "vld1.s8 {d2}, [%[inptr4]], r1 \n" // d2=e0e1e2e3e4e5e6e7
            "vld1.s8 {d6}, [%[inptr5]], r1 \n" // d6=f0f1f2f3f4f5f6f7
            "vld1.s8 {d3}, [%[inptr6]], r1 \n" // d3=g0g1g2g3g4g5g6g7
            "vld1.s8 {d7}, [%[inptr7]], r1 \n" // d7=h0h1h2h3h4h5h6h7

            "vtrn.32  q0, q2 \n"  //q0=a0~a3b0~b3c0~c3d0~d3
            "vtrn.32  q1, q3 \n"  //q1=e0~e3f0~f3g0~g3h0~h3

            "vst1.32 {d0,d1,d2,d3}, [%[outptr_row0]]! \n"
            : [inptr0] "+r"(inptr_row[0]), [inptr1] "+r"(inptr_row[1]),
            [inptr2] "+r"(inptr_row[2]), [inptr3] "+r"(inptr_row[3]),
             [inptr4] "+r"(inptr_row[4]), [inptr5] "+r"(inptr_row[5]),
             [inptr6] "+r"(inptr_row[6]), [inptr7] "+r"(inptr_row[7]),
              [outptr_row0] "+r"(out0)
            :
            : "r1", "q0", "q1", "q2", "q3", "cc", "memory"
            );
            right_remain -= 4;
        }
    // clang-format on
    if (right_remain > 0) {
      for (int i = 0; i < 8; i++) {
        for (int x = 0; x < right_remain; x++) {
          *out0++ = *inptr_row[i]++;
        }
        for (int x = 0; x < 4 - right_remain; x++) {
          *out0++ = 0;
        }
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
}

template <typename Dtype>
void gemm_prepack_vsdot_int8(const int8_t* A_packed,
                             const int8_t* B,
                             const float* bias,
                             Dtype* C,
                             int M,
                             int N,
                             int K,
                             bool is_bias,
                             int is_relu,
                             bool is_transB,
                             const float* scale,
                             const float* alpha,
                             ARMContext* ctx) {
  size_t llc_size = ctx->llc_size() / 4;
  auto workspace = ctx->workspace_data<int8_t>();
  int x_block = (llc_size - (MBLOCK_INT8_DOT * K)) /
                (sizeof(int8_t) * (K + MBLOCK_INT8_DOT));
  x_block /= NBLOCK_INT8_DOT;
  x_block *= NBLOCK_INT8_DOT;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK_INT8_DOT - 1) / NBLOCK_INT8_DOT;
  x_block *= NBLOCK_INT8_DOT;
  x_block = x_block < NBLOCK_INT8_DOT ? NBLOCK_INT8_DOT : x_block;
  int kup = ROUNDUP(K, KBLOCK_INT8);
  int tail_pre = ((kup / 4) & (KBLOCK_INT8 - 1));
  int k = (kup / 4);

  bool flag_p_remain = false;
  int remain = 0;

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK_INT8_DOT - 1) / NBLOCK_INT8_DOT;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK_INT8_DOT;
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    auto b_pannel = static_cast<int8_t*>(workspace);
    if (!is_transB) {
      // K * N
      packb_dot_int8(b_pannel, B, N, 0, K, x0, xmax);
    } else {
      // N X K
      packb_dot_trans_int8(b_pannel, B, K, 0, K, x0, xmax);
    }
    LITE_PARALLEL_COMMON_BEGIN(y, tid, M, 0, MBLOCK_INT8_DOT) {
      unsigned int ymax = y + MBLOCK_INT8_DOT;
      if (ymax > M) {
        ymax = M;
      }

      float32_t bias_local[6] = {0, 0, 0, 0, 0, 0};
      if (is_bias) {
        int j = 0;
        for (int i = y; i < ymax && j < 6; i++, j++) {
          bias_local[j] = bias[i];
        }
      }
      float32_t scale_local[6];
      if (scale) {
        int j = 0;
        for (int i = y; i < ymax && j < 6; i++, j++) {
          scale_local[j] = scale[i];
        }
      }

      Dtype cout0[NBLOCK_INT8_DOT];
      Dtype cout1[NBLOCK_INT8_DOT];
      Dtype cout2[NBLOCK_INT8_DOT];
      Dtype cout3[NBLOCK_INT8_DOT];
      Dtype cout4[NBLOCK_INT8_DOT];
      Dtype cout5[NBLOCK_INT8_DOT];

      Dtype* c_ptr0 = C + y * N + x0;
      Dtype* c_ptr1 = c_ptr0 + N;
      Dtype* c_ptr2 = c_ptr1 + N;
      Dtype* c_ptr3 = c_ptr2 + N;
      Dtype* c_ptr4 = c_ptr3 + N;
      Dtype* c_ptr5 = c_ptr4 + N;

      Dtype* pout0 = c_ptr0;
      Dtype* pout1 = c_ptr1;
      Dtype* pout2 = c_ptr2;
      Dtype* pout3 = c_ptr3;
      Dtype* pout4 = c_ptr4;
      Dtype* pout5 = c_ptr5;

      const int8_t* a_ptr_l = A_packed + y * kup;
      const int8_t* b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if ((y + 5) >= ymax) {
          switch ((y + 5) - ymax) {
            case 4:
              c_ptr1 = cout1;
            case 3:
              c_ptr2 = cout2;
            case 2:
              c_ptr3 = cout3;
            case 1:
              c_ptr4 = cout4;
            case 0:
              c_ptr5 = cout5;
            default:
              break;
          }
        }
        if (flag_p_remain && (xb == bblocks - 1)) {
          pout0 = c_ptr0;
          pout1 = c_ptr1;
          pout2 = c_ptr2;
          pout3 = c_ptr3;
          pout4 = c_ptr4;
          pout5 = c_ptr5;

          c_ptr0 = cout0;
          c_ptr1 = cout1;
          c_ptr2 = cout2;
          c_ptr3 = cout3;
          c_ptr4 = cout4;
          c_ptr5 = cout5;
        }
        const int8_t* a_ptr = a_ptr_l;
        int tail = tail_pre;

        float32_t* scale_ptr = scale_local;
        float32_t* bias_ptr = bias_local;

        gemm_dot_int8_kernel<Dtype>(a_ptr,
                                    b_ptr,
                                    bias_ptr,
                                    c_ptr0,
                                    c_ptr1,
                                    c_ptr2,
                                    c_ptr3,
                                    c_ptr4,
                                    c_ptr5,
                                    scale_ptr,
                                    alpha,
                                    is_relu,
                                    k,
                                    tail);
        scale_ptr = scale_local;
        bias_ptr = bias_local;

        if (flag_p_remain && (xb == bblocks - 1)) {
          for (int i = 0; i < remain; ++i) {
            *pout0++ = cout0[i];
            *pout1++ = cout1[i];
            *pout2++ = cout2[i];
            *pout3++ = cout3[i];
            *pout4++ = cout4[i];
            *pout5++ = cout5[i];
          }
        }
      }
    }
    LITE_PARALLEL_COMMON_END();
  }
}
#endif
#endif  // dotprod  //NOLINT

template <typename dtype>
void gemm_prepack_int8(const int8_t* A_packed,
                       const int8_t* B,
                       const float* bias,
                       dtype* C,
                       int M,
                       int N,
                       int K,
                       bool is_bias,
                       bool is_transB,
                       const float* scale,
                       const operators::ActivationParam act_param,
                       ARMContext* ctx) {
  auto act_type = act_param.active_type;
  float alpha[12] = {
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 0x04;
      for (int i = 0; i < 4; i++) {
        alpha[i] = 1.f / act_param.hard_swish_scale;
        alpha[i + 4] = act_param.hard_swish_offset;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }

#define IN_PARAMS \
  A_packed, B, bias, C, M, N, K, is_bias, flag_act, is_transB, scale, alpha, ctx
#ifdef __aarch64__
  if (ctx->has_dot()) {
#ifdef WITH_ARM_DOTPROD
    gemm_prepack_sdot_int8<dtype>(IN_PARAMS);
#endif
  } else {
    gemm_prepack_oth_int8<dtype>(IN_PARAMS);
  }
#else
  if (ctx->has_dot()) {
#ifdef WITH_ARM_DOTPROD
    gemm_prepack_vsdot_int8<dtype>(IN_PARAMS);
#endif
  } else {
    gemm_prepack_oth_int8<dtype>(IN_PARAMS);
  }
#endif
}

#define GEMM_PREPACK_INT8(dtype)                  \
  template void gemm_prepack_int8<dtype>(         \
      const int8_t* A_packed,                     \
      const int8_t* B,                            \
      const float* bias,                          \
      dtype* C,                                   \
      int M,                                      \
      int N,                                      \
      int K,                                      \
      bool is_bias,                               \
      bool is_transB,                             \
      const float* scale,                         \
      const operators::ActivationParam act_param, \
      ARMContext* ctx);
GEMM_PREPACK_INT8(int8_t);
GEMM_PREPACK_INT8(float_t);
GEMM_PREPACK_INT8(int32_t);
#undef IN_PARAMS
#undef GEMM_PREPACK_INT8

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
