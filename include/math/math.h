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
#ifndef MDL_MATH_H
#define MDL_MATH_H

#include "commons/commons.h"
#include <array>
#include <cmath>

namespace mdl {
#ifdef ANDROID
    const std::array<float32x4_t, 8> exp_tab = {
        {
            vdupq_n_f32(1.f),
            vdupq_n_f32(0.0416598916054f),
            vdupq_n_f32(0.500000596046f),
            vdupq_n_f32(0.0014122662833f),
            vdupq_n_f32(1.00000011921f),
            vdupq_n_f32(0.00833693705499f),
            vdupq_n_f32(0.166665703058f),
            vdupq_n_f32(0.000195780929062f),
        }
    };
    
    const std::array<float32x4_t, 8> log_tab = {
        {
            vdupq_n_f32(-2.29561495781f),
            vdupq_n_f32(-2.47071170807f),
            vdupq_n_f32(-5.68692588806f),
            vdupq_n_f32(-0.165253549814f),
            vdupq_n_f32(5.17591238022f),
            vdupq_n_f32(0.844007015228f),
            vdupq_n_f32(4.58445882797f),
            vdupq_n_f32(0.0141278216615f),
        }
    };
    
    inline float32x4_t vinvsqrt_f32(float32x4_t x) {
        float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);
        sqrt_reciprocal             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
        sqrt_reciprocal             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
        return sqrt_reciprocal;
    }
    
    inline float32x4_t vinv_f32(const float32x4_t &x) {
        float32x4_t recip = vrecpeq_f32(x);
        recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
        recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
        return recip;
    }
    
    inline float32x4_t vtaylor_poly_f32(const float32x4_t &x, const std::array<float32x4_t, 8> &coeffs) {
        float32x4_t A   = vmlaq_f32(coeffs[0], coeffs[4], x);
        float32x4_t B   = vmlaq_f32(coeffs[2], coeffs[6], x);
        float32x4_t C   = vmlaq_f32(coeffs[1], coeffs[5], x);
        float32x4_t D   = vmlaq_f32(coeffs[3], coeffs[7], x);
        float32x4_t x2  = vmulq_f32(x, x);
        float32x4_t x4  = vmulq_f32(x2, x2);
        float32x4_t res = vmlaq_f32(vmlaq_f32(A, B, x2), vmlaq_f32(C, D, x2), x4);
        return res;
    }
    
    inline float32x4_t vexp_f32(const float32x4_t &x) {
        static const float32x4_t CONST_LN2     = vdupq_n_f32(0.6931471805f);
        static const float32x4_t CONST_INV_LN2 = vdupq_n_f32(1.4426950408f);
    
        int32x4_t   m   = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
        float32x4_t val = vmlsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);
    
        float32x4_t poly = vtaylor_poly_f32(val, exp_tab);
        poly = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(poly), vshlq_n_s32(m, 23)));
    
        return poly;
    }
    
    inline float32x4_t vlog_f32(const float32x4_t &x) {
        static const int32x4_t   CONST_127 = vdupq_n_s32(127);
        static const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f);
    
        int32x4_t   m   = vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23)), CONST_127);
        float32x4_t val = vreinterpretq_f32_s32(vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));
    
        float32x4_t poly = vtaylor_poly_f32(val, log_tab);
        poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);
    
        return poly;
    }
    
    inline float32x4_t vtanh_f32(const float32x4_t &val) {
        static const float32x4_t CONST_1 = vdupq_n_f32(1.f);
        static const float32x4_t CONST_2 = vdupq_n_f32(2.f);
    
        float32x4_t exp2x = vexp_f32(vmulq_f32(CONST_2, val));
        float32x4_t num   = vsubq_f32(exp2x, CONST_1);
        float32x4_t den   = vaddq_f32(exp2x, CONST_1);
        float32x4_t tanh  = vmulq_f32(num, vinv_f32(den));
    
        return tanh;
    }
    
    inline float32x4_t vpowq_f32(const float32x4_t &val, const float32x4_t &n) {
        return vexp_f32(vmulq_f32(n, vlog_f32(val)));
    }
#endif

    struct Math {
        inline static void v_sqr(const int n, const float *xs, float *ys) {
#ifdef ANDROID
            int i, m = n / 4;
            float32x4_t xv, yv;
            for (i = 0; i < m; i++) {
                xv = vld1q_f32(xs + i * 4);
                yv = vmulq_f32(xv, xv);
                vst1q_f32(ys + i * 4, yv);
            }
            for (i = 4 * m; i < n; i++) {
                ys[i] = xs[i] * xs[i];
            }
#else
            for (int i = 0; i < n; i++) {
                ys[i] = xs[i] * xs[i];
            }
#endif
        }

        inline static void axpy(const int n, const float alpha, const float *xs, float *ys) {
#ifdef ANDROID
            int i, m = n / 4;
            float32x4_t xv, yv, av = vdupq_n_f32(alpha);
            for (i = 0; i < m; i++) {
                xv = vld1q_f32(xs + i * 4);
                yv = vld1q_f32(ys + i * 4);
                yv = vmlaq_f32(yv, av, xv);
                vst1q_f32(ys + i * 4, yv);
            }
            for (i = 4 * m; i < n; i++) {
                ys[i] = alpha * xs[i] + ys[i];
            }
#else
            for (int i = 0; i < n; i++) {
                ys[i] = alpha * xs[i] + ys[i];
            }
#endif
        }

        inline static void v_pow(const int n, const float *xs, const float p, float *ys) {
#ifdef ANDROID
            int i, m = n / 4;
            float32x4_t xv, yv, pv = vdupq_n_f32(p);
            for (i = 0; i < m; i++) {
                xv = vld1q_f32(xs + i * 4);
                yv = vpowq_f32(xv, pv);
                vst1q_f32(ys + i * 4, yv);
            }
            for (i = 4 * m; i < n; i++) {
                ys[i] = pow(xs[i], p);
            }
#else
            for (int i = 0; i < n; i++) {
                ys[i] = pow(xs[i], p);
            }
#endif
        }

        inline static void v_mul(const int n, const float *xs, const float *ys, float *zs) {
#ifdef ANDROID
            int i, m = n / 4;
            float32x4_t xv, yv, zv;
            for (i = 0; i < m; i++) {
                xv = vld1q_f32(xs + i * 4);
                yv = vld1q_f32(ys + i * 4);
                zv = vmulq_f32(xv, yv);
                vst1q_f32(zs + i * 4, zv);
            }
            for (i = 4 * m; i < n; i++) {
                zs[i] = xs[i] * ys[i];
            }
#else
            for (int i = 0; i < n; i++) {
                zs[i] = xs[i] * ys[i];
            }
#endif
        }

        // static void transpose(const int m, const int n, float *xs) {
        //     float *trans_data = new float[m * n];
        //     for (int i = 0; i < m; i++) {
        //         for (int j = 0; j < n; j++) {
        //             trans_data[j * m + i] = xs[i * n + j];
        //         }
        //     }
        //     for (int i = 0; i < n; i++) {
        //         for (int j = 0; j < m; j++) {
        //             xs[i * m + j] = trans_data[i * m + j];
        //         }
        //     }
        //     delete[] trans_data;
        // }

        inline static void v_scale(const int n, const float scale_factor, const float *data, float *dest) {
            for (int i = 0; i < n; ++i) {
                dest[i] = data[i] * scale_factor;

            }

        }

        inline static void v_add(const int N, const float alpha, float *Y) {
            for (int i = 0; i < N; ++i) {
                Y[i] += alpha;

            }
        }

        inline static void v_div(const int n, const float *a, const float *b, float *y) {
            for (int i = 0; i < n; ++i) {
                if (b[i] != 0) {
                    y[i] = a[i] / b[i];
                }
            }
        }

        inline static void v_exp(const int n, const float *a, float *y) {

            for (int i = 0; i < n; ++i) {
                    y[i] = expf(a[i]);

            }


        }
    };
};

#endif
