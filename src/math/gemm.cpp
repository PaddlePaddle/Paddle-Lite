/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/
#include "math/gemm.h"

namespace mdl {
    vector<Gemmer *> Gemmer::gemmers;

    void Gemmer::pack_MRxk(int k, const float *A, int incRowA, int incColA, float *buffer) {
        int j, a2 = incRowA, a3 = 2 * incRowA, a4 = 3 * incRowA;
        for (j = 0; j < k; ++j) {
            // for (int i = 0; i < MR; ++i) {
            //     buffer[i] = A[i * incRowA];
            // }
            buffer[0] = A[0];
            buffer[1] = A[a2];
            buffer[2] = A[a3];
            buffer[3] = A[a4];
            A += 1;
            buffer += MR;
        }
    }

    void Gemmer::pack_A(int mc, int kc, const float *A, int incRowA, int incColA, float *buffer) {
        int mp = mc / MR;
        int _mr = mc % MR;
        int tmp1 = kc * MR;
        int tmp2 = MR * incRowA;
        int i, j;

        for (i = 0; i < mp; ++i) {
            pack_MRxk(kc, A, incRowA, incColA, buffer);
            buffer += tmp1;
            A += tmp2;
            // buffer += kc * MR;
            // A += MR * incRowA;
        }
        if (_mr > 0) {
            for (j = 0; j < kc; ++j) {
                for (i = 0; i < _mr; ++i) {
                    buffer[i] = A[i * incRowA];
                }
                for (i = _mr; i < MR; ++i) {
                    buffer[i] = 0.0;
                }
                A += 1;
                buffer += MR;
            }
        }
    }

    void Gemmer::pack_kxNR(int k, const float *B, int incRowB, int incColB, float *buffer) {
        int i, j;
        for (i = 0; i < k; ++i) {
            for (j = 0; j < NR; ++j) {
                buffer[j] = B[j];
            }
            // float32x4_t bv = vld1q_f32(B);
            // vst1q_f32(buffer, bv);
            B += incRowB;
            buffer += NR;
        }
    }

    void Gemmer::pack_B(int kc, int nc, const float *B, int incRowB, int incColB, float *buffer) {
        int np = nc / NR;
        int _nr = nc % NR;
        int tmp1 = kc * NR;
        int i, j;

        for (j = 0; j < np; ++j) {
            pack_kxNR(kc, B, incRowB, incColB, buffer);
            B += NR;
            buffer += tmp1;
        }
        if (_nr > 0) {
            for (i = 0; i < kc; ++i) {
                for (j = 0; j < _nr; ++j) {
                    buffer[j] = B[j];
                }
                for (j = _nr; j < NR; ++j) {
                    buffer[j] = 0.0;
                }
                buffer += NR;
                B += incRowB;
            }
        }
    }
    
#if defined(MDL_V7_iOS)
    void Gemmer::dgemm_micro_kernel(int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC,
                                    int incColC) {
        int i, j, l;
        float32x4_t abv0 = vdupq_n_f32(0);
        float32x4_t abv1 = vdupq_n_f32(0);
        float32x4_t abv2 = vdupq_n_f32(0);
        float32x4_t abv3 = vdupq_n_f32(0);
        
        float32x4_t av;
        float32x4_t bv;
        
        float32x2_t bv01;
        float32x2_t bv23;
        
        for (l = 0; l < kc; ++l) {
            av = vld1q_f32(A);
            bv = vld1q_f32(B);
            bv01 = vget_low_f32(bv);
            abv0 = vmlaq_lane_f32(abv0, av, bv01, 0);
            abv1 = vmlaq_lane_f32(abv1, av, bv01, 1);
            bv23 = vget_high_f32(bv);
            abv2 = vmlaq_lane_f32(abv2, av, bv23, 0);
            abv3 = vmlaq_lane_f32(abv3, av, bv23, 1);
            A += MR;
            B += NR;
        }
        
        vst1q_f32(AB_ + 0, abv0);
        vst1q_f32(AB_ + 4, abv1);
        vst1q_f32(AB_ + 8, abv2);
        vst1q_f32(AB_ + 12, abv3);
        
        if (equal(beta, 0.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = 0.0;
                }
            }
        } else if (!equal(beta, 1.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] *= beta;
                }
            }
        }
        
        if (!equal(alpha, 1.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] += alpha * AB_[i + j * MR];
                }
            }
        } else {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] += AB_[i + j * MR];
                }
            }
        }
    }
#elif defined(MDL_V7)
    void Gemmer::dgemm_micro_kernel(int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC,
                               int incColC) {

        int kc1 = kc / 2, kc2 = kc % 2;

        asm volatile(
            "vmov.f32   q10,    #0.0        \n\t"
            "vmov.f32   q11,    #0.0        \n\t"
            "vmov.f32   q12,    #0.0        \n\t"
            "vmov.f32   q13,    #0.0        \n\t"
            "subs       %[kc1], %[kc1], #1  \n\t"
            "blt        end_kc1_%=          \n\t"
            "loop_kc1_%=:                   \n\t"
            "pld        [%[B], #256]        \n\t"
            "pld        [%[A], #256]        \n\t"
            "vld1.32    {q0, q1}, [%[B]]!   \n\t"
            "vld1.32    {q2, q3}, [%[A]]!   \n\t"
            "vmla.f32   q10, q2, d0[0]      \n\t"
            "vmla.f32   q11, q2, d0[1]      \n\t"
            "vmla.f32   q12, q2, d1[0]      \n\t"
            "vmla.f32   q13, q2, d1[1]      \n\t"

            "vmla.f32   q10, q3, d2[0]      \n\t"
            "vmla.f32   q11, q3, d2[1]      \n\t"
            "vmla.f32   q12, q3, d3[0]      \n\t"
            "vmla.f32   q13, q3, d3[1]      \n\t"

            "subs       %[kc1], %[kc1], #1  \n\t"
            "bge        loop_kc1_%=         \n\t"
            "end_kc1_%=:                    \n\t"

            "subs       %[kc2], %[kc2], #1  \n\t"
            "blt        end_kc2_%=          \n\t"
            "loop_kc2_%=:                   \n\t"
            "vld1.32    {q4}, [%[B]]!       \n\t"
            "vld1.32    {q5}, [%[A]]!       \n\t"
            "vmla.f32   q10, q5, d8[0]      \n\t"
            "vmla.f32   q11, q5, d8[1]      \n\t"
            "vmla.f32   q12, q5, d9[0]      \n\t"
            "vmla.f32   q13, q5, d9[1]      \n\t"
            "subs       %[kc2], %[kc2], #1  \n\t"
            "bge        loop_kc2_%=         \n\t"
            "end_kc2_%=:                    \n\t"

            "vst1.32    {q10, q11}, [%[AB_]]!    \n\t"
            "vst1.32    {q12, q13}, [%[AB_]]!    \n\t"
            :
            :[A]"r"(A), [B]"r"(B), [kc1]"r"(kc1), [kc2]"r"(kc2), [AB_]"r"(AB_)
            :"memory", "q0", "q1", "q2", "q3", "q4", "q5", "q10", "q11", "q12", "q13"
        );

        int i, j;

        if (equal(beta, 0.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = 0.0;
                }
            }
        } else if (!equal(beta, 1.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] *= beta;
                }
            }
        }

        if (!equal(alpha, 1.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] += alpha * AB_[i + j * MR];
                }
            }
        } else {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] += AB_[i + j * MR];
                }
            }
        }
    }
#elif defined(MDL_V8)
    void Gemmer::dgemm_micro_kernel(int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC) {
            int i, j, l;

            float32x4_t abv0 = vdupq_n_f32(0);
            float32x4_t abv1 = vdupq_n_f32(0);
            float32x4_t abv2 = vdupq_n_f32(0);
            float32x4_t abv3 = vdupq_n_f32(0);

            float32x4_t av;
            float32x4_t bv;

            int kc1 = kc / 4, kc2 = kc % 4;
            asm volatile (
            "subs %[kc1], %[kc1], #1\n\t"
                    "blt end1\n\t"
                    "loop1: \n\t"
                    "ld1 {%[av].4S}, [%[A]], #16\n\t"
                    "ld1 {%[bv].4S}, [%[B]], #16\n\t"
                    "fmla %[abv0].4S, %[av].4S, %[bv].4S[0]\n\t"
                    "fmla %[abv1].4S, %[av].4S, %[bv].4S[1]\n\t"
                    "fmla %[abv2].4S, %[av].4S, %[bv].4S[2]\n\t"
                    "fmla %[abv3].4S, %[av].4S, %[bv].4S[3]\n\t"
                    // "add %[A], %[A], #16\n\t"
                    // "add %[B], %[B], #16\n\t"
                    "ld1 {%[av].4S}, [%[A]], #16\n\t"
                    "ld1 {%[bv].4S}, [%[B]], #16\n\t"
                    "fmla %[abv0].4S, %[av].4S, %[bv].4S[0]\n\t"
                    "fmla %[abv1].4S, %[av].4S, %[bv].4S[1]\n\t"
                    "fmla %[abv2].4S, %[av].4S, %[bv].4S[2]\n\t"
                    "fmla %[abv3].4S, %[av].4S, %[bv].4S[3]\n\t"
                    // "add %[A], %[A], #16\n\t"
                    // "add %[B], %[B], #16\n\t"
                    "ld1 {%[av].4S}, [%[A]], #16\n\t"
                    "ld1 {%[bv].4S}, [%[B]], #16\n\t"
                    "fmla %[abv0].4S, %[av].4S, %[bv].4S[0]\n\t"
                    "fmla %[abv1].4S, %[av].4S, %[bv].4S[1]\n\t"
                    "fmla %[abv2].4S, %[av].4S, %[bv].4S[2]\n\t"
                    "fmla %[abv3].4S, %[av].4S, %[bv].4S[3]\n\t"
                    // "add %[A], %[A], #16\n\t"
                    // "add %[B], %[B], #16\n\t"
                    "ld1 {%[av].4S}, [%[A]], #16\n\t"
                    "ld1 {%[bv].4S}, [%[B]], #16\n\t"
                    "fmla %[abv0].4S, %[av].4S, %[bv].4S[0]\n\t"
                    "fmla %[abv1].4S, %[av].4S, %[bv].4S[1]\n\t"
                    "fmla %[abv2].4S, %[av].4S, %[bv].4S[2]\n\t"
                    "fmla %[abv3].4S, %[av].4S, %[bv].4S[3]\n\t"
                    // "add %[A], %[A], #16\n\t"
                    // "add %[B], %[B], #16\n\t"
                    "subs %[kc1], %[kc1], #1\n\t"
                    "bge loop1\n\t"
                    "end1:\n\t"
                    "subs %[kc2], %[kc2], #1\n\t"
                    "blt end2\n\t"
                    "loop2: \n\t"
                    "ld1 {%[av].4S}, [%[A]]\n\t"
                    "ld1 {%[bv].4S}, [%[B]]\n\t"
                    "fmla %[abv0].4S, %[av].4S, %[bv].4S[0]\n\t"
                    "fmla %[abv1].4S, %[av].4S, %[bv].4S[1]\n\t"
                    "fmla %[abv2].4S, %[av].4S, %[bv].4S[2]\n\t"
                    "fmla %[abv3].4S, %[av].4S, %[bv].4S[3]\n\t"
                    "add %[A], %[A], #16\n\t"
                    "add %[B], %[B], #16\n\t"
                    "subs %[kc2], %[kc2], #1\n\t"
                    "bge loop2\n\t"
                    "end2:\n\t"
            : [A]"=r"(A), [B]"=r"(B), [av]"=w"(av), [bv]"=w"(bv),
            [abv0]"=w"(abv0), [abv1]"=w"(abv1), [abv2]"=w"(abv2), [abv3]"=w"(abv3),
            [kc1]"=r"(kc1), [kc2]"=r"(kc2)
            : "[A]"(A), "[B]"(B), "[av]"(av), "[bv]"(bv),
                    "[abv0]"(abv0), "[abv1]"(abv1), "[abv2]"(abv2), "[abv3]"(abv3),
                    "[kc1]"(kc1), "[kc2]"(kc2)
            );

            vst1q_f32(AB_ + 0, abv0);
            vst1q_f32(AB_ + 4, abv1);
            vst1q_f32(AB_ + 8, abv2);
            vst1q_f32(AB_ + 12, abv3);

            if (beta == 0.0) {
                for (j = 0; j < NR; ++j) {
                    for (i = 0; i < MR; ++i) {
                        C[i * incRowC + j * incColC] = 0.0;
                    }
                }
            } else if (beta != 1.0) {
                for (j = 0; j < NR; ++j) {
                    for (i = 0; i < MR; ++i) {
                        C[i * incRowC + j * incColC] *= beta;
                    }
                }
            }

            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] += AB_[i + j * MR];
                }
            }
        }


#else
    void Gemmer::dgemm_micro_kernel(int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC,
                               int incColC) {
        int i = 0;
        int j = 0;
        int l = 0;
        for (l = 0; l < MR * NR; ++l) {
            AB_[l] = 0;
        }
        for (l = 0; l < kc; ++l) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    AB_[i + j * MR] += A[i] * B[j];
                }
            }
            A += MR;
            B += NR;
        }

        if (equal(beta, 0.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = 0.0;
                }
            }
        } else if (!equal(beta, 1.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] *= beta;
                }
            }
        }

        if (!equal(alpha, 1.0)) {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] += alpha * AB_[i + j * MR];
                }
            }
        } else {
            for (j = 0; j < NR; ++j) {
                for (i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] += AB_[i + j * MR];
                }
            }
        }
    }
#endif


    void Gemmer::dgeaxpy(int m, int n, float alpha, const float *X, int incRowX, int incColX, float *Y, int incRowY,
                         int incColY) {
        int i, j;
        if (!equal(alpha, 1.0)) {
            for (j = 0; j < n; ++j) {
                for (i = 0; i < m; ++i) {
                    Y[i * incRowY + j] += alpha * X[i + j * incColX];
                }
            }
        } else {
            for (j = 0; j < n; ++j) {
                for (i = 0; i < m; ++i) {
                    Y[i * incRowY + j] += X[i + j * incColX];
                }
            }
        }
    }

    void Gemmer::dgescal(int m, int n, float alpha, float *X, int incRowX, int incColX) {
        int i, j;
        if (!equal(alpha, 0.0)) {
            for (i = 0; i < m; ++i) {
                for (j = 0; j < n; ++j) {
                    X[i * incRowX + j] *= alpha;
                }
            }
        } else {
            for (i = 0; i < m; ++i) {
                for (j = 0; j < n; ++j) {
                    X[i * incRowX + j] = 0.0;
                }
            }
        }
    }

    void Gemmer::dgemm_macro_kernel(int mc, int nc, int kc, float alpha, float beta, float *C, int incRowC, int incColC) {
        int mp = (mc + MR - 1) / MR;
        int np = (nc + NR - 1) / NR;

        int _mr = mc % MR;
        int _nr = nc % NR;

        int i, j;

        for (j = 0; j < np; ++j) {
            int nr = (j != np - 1 || _nr == 0) ? NR : _nr;

            for (i = 0; i < mp; ++i) {
                int mr = (i != mp - 1 || _mr == 0) ? MR : _mr;

                if (mr == MR && nr == NR) {
                    dgemm_micro_kernel(kc, alpha, &A_[i * kc * MR], &B_[j * kc * NR], beta, &C[i * MR * incRowC + j * NR], incRowC, incColC);
                } else {
                    dgemm_micro_kernel(kc, alpha, &A_[i * kc * MR], &B_[j * kc * NR], 0.0, C_, 1, MR);
                    dgescal(mr, nr, beta, &C[i * MR * incRowC + j * NR], incRowC, incColC);
                    dgeaxpy(mr, nr, 1.0, C_, 1, MR, &C[i * MR * incRowC + j * NR], incRowC, incColC);
                }
            }
        }
    }

    void Gemmer::dgemm_nn(int m, int n, int k, float alpha, const float *A, int incRowA, int incColA, const float *B, int incRowB, int incColB, float beta, float *C, int incRowC, int incColC) {
        int mb = (m + MC - 1) / MC;
        int nb = (n + NC - 1) / NC;
        int kb = (k + KC - 1) / KC;

        int _mc = m % MC;
        int _nc = n % NC;
        int _kc = k % KC;

        int mc, nc, kc;
        int i, j, l;

        float _beta;

        if (equal(alpha, 0.0) ||  k == 0) {
            dgescal(m, n, beta, C, incRowC, incColC);
            return;
        }

        for (j = 0; j < nb; ++j) {
            nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

            for (l = 0; l < kb; ++l) {
                kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
                _beta = (l == 0) ? beta : 1.0;

                pack_B(kc, nc, &B[l * KC * incRowB + j * NC], incRowB, incColB, B_);

                for (i = 0; i < mb; ++i) {
                    mc = (i != mb - 1 || _mc == 0) ? MC : _mc;

                    pack_A(mc, kc, &A[i * MC * incRowA + l * KC], incRowA, incColA, A_);

                    dgemm_macro_kernel(mc, nc, kc, alpha, _beta, &C[i * MC * incRowC + j * NC], incRowC, incColC);
                }
            }
        }
    }

    void Gemmer::sgemm(int m, int n, int k, const float *A, const float *B, float *C) {
        dgemm_nn(m, n, k, 1, A, k, 1, B, n, 1, 0, C, n, 1);
    }

    void Gemmer::sgemm(int m, int n, int k, const float *A, const float *B, float *C, float alpha, float beta) {
        dgemm_nn(m, n, k, alpha, A, k, 1, B, n, 1, beta, C, n, 1);
    }
};
