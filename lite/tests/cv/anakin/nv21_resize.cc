#include <limits.h>
#include <math.h>
#include "lite/tests/cv/anakin/cv_utils.h"
void resize_one_channel(const unsigned char* src, int w_in, int h_in, unsigned char* dst, int w_out, int h_out);
void resize_one_channel_uv(const unsigned char* src, int w_in, int h_in, unsigned char* dst, int w_out, int h_out);
void nv21_resize(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out){
    if (w_out == w_in && h_out == h_in)
    {
        printf("nv21_resize equal \n");
        memcpy(dst, src, sizeof(unsigned char) * w_in * (int)(1.5 * h_in));
        return;
    }
   // printf("w_in: %d, h_in: %d, w_out: %d, h_out: %d\n", w_in, h_in, w_out, h_out);
   // dst = new unsigned char[h_out * w_out];
    //if (dst == nullptr)
   //     return;
    int y_h = h_in;
    int uv_h =  h_in / 2;
    const unsigned char* y_ptr = src;
    const unsigned char* uv_ptr = src + y_h * w_in;
    //out
    int dst_y_h = h_out;
    int dst_uv_h = h_out / 2;
    unsigned char* dst_ptr = dst + dst_y_h * w_out;

    //resize_one_channel(src, w_in, h_in, dst, w_out, h_out);
    //printf("resize_one_channel dst_y_h: %d,  y_ptr: %x, dst： %x \n", dst_y_h, y_ptr, dst);
    //y
    resize_one_channel(y_ptr, w_in, y_h, dst, w_out, dst_y_h);
   //printf("resize_one_channel_uv dst_uv_h: %d, uv_ptr: %x, dst_uv: %x \n", dst_uv_h, uv_ptr, dst_ptr);
    //uv
    resize_one_channel_uv(uv_ptr, w_in, uv_h, dst_ptr, w_out, dst_uv_h);
}

void resize_one_channel(const unsigned char* src, int w_in, int h_in, unsigned char* dst, int w_out, int h_out){
    //printf("resize_one_channel \n");
    const int resize_coef_bits = 11;
    const int resize_coef_scale = 1 << resize_coef_bits;

    double scale_x = (double)w_in / w_out;
    double scale_y = (double)h_in / h_out;

    int* buf = new int[w_out * 2 + h_out * 2];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w_out;//new int[h];

    short* ialpha = (short*)(buf + w_out + h_out);//new short[w * 2];
    short* ibeta = (short*)(buf + w_out * 2 + h_out);//new short[h * 2];

    float fx = 0.f;
    float fy = 0.f;
    int sx = 0;
    int sy = 0;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);
// #pragma omp parallel for
    for (int dx = 0; dx < w_out; dx++){
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = floor(fx);
        fx -= sx;

        if (sx < 0){
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w_in - 1){
            sx = w_in - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        float a0 = (1.f - fx) * resize_coef_scale;
        float a1 =        fx  * resize_coef_scale;

        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }
// #pragma omp parallel for
    for (int dy = 0; dy < h_out; dy++) {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = floor(fy);
        fy -= sy;

        if (sy < 0){
            sy = 0;
            fy = 0.f;
        }
        if (sy >= h_in - 1){
            sy = h_in - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * resize_coef_scale;
        float b1 =        fy  * resize_coef_scale;

        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }
#undef SATURATE_CAST_SHORT
    // loop body
    short* rowsbuf0 = new short[w_out + 1];
    short* rowsbuf1 = new short[w_out + 1];
    short* rows0 = rowsbuf0;
    short* rows1 = rowsbuf1;

    int prev_sy1 = -1;
    for (int dy = 0; dy < h_out; dy++){
        int sy = yofs[dy];

        if (sy == prev_sy1){
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + w_in * (sy + 1);
            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w_out; dx++){
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
                rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialphap += 2;
            }
        }else{
            // hresize two rows
            const unsigned char *S0 = src + w_in * (sy);
            const unsigned char *S1 = src + w_in * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w_out; dx++){
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
                rows0p[dx] = (S0p[0] * a0 + S0p[1] * a1) >> 4;
                rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialphap += 2;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* dp_ptr = dst + w_out * (dy);

        int cnt = w_out >> 3;
        int remain = w_out - (cnt << 3);
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);

// #pragma omp parallel for

#if 1 //__aarch64__
        for (cnt = w_out >> 3; cnt > 0; cnt--){
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _dout = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(dp_ptr, _dout);

            dp_ptr += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
#pragma omp parallel for
        if (cnt > 0){
        asm volatile(
            "mov        r4, #2          \n"
            "vdup.s32   q12, r4         \n"
             "0:                         \n"
            "pld        [%[rows0p], #128]      \n"
            "pld        [%[rows1p], #128]      \n"
            "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
            "vld1.s16   {d6-d7}, [%[rows0p]]!\n"
            "pld        [%[rows0p], #128]      \n"
            "pld        [%[rows1p], #128]      \n"
            "vmull.s16  q0, d2, %[_b0]     \n"
            "vmull.s16  q1, d3, %[_b0]     \n"
            "vmull.s16  q2, d6, %[_b1]     \n"
            "vmull.s16  q3, d7, %[_b1]     \n"

            "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
            "vld1.s16   {d6-d7}, [%[rows0p]]!\n"

            "vorr.s32   q10, q12, q12   \n"
            "vorr.s32   q11, q12, q12   \n"
            "vsra.s32   q10, q0, #16    \n"
            "vsra.s32   q11, q1, #16    \n"
            "vsra.s32   q10, q2, #16    \n"
            "vsra.s32   q11, q3, #16    \n"

            "vmull.s16  q0, d2, %[_b0]     \n"
            "vmull.s16  q1, d3, %[_b0]     \n"
            "vmull.s16  q2, d6, %[_b1]     \n"
            "vmull.s16  q3, d7, %[_b1]     \n"

            "vsra.s32   q10, q0, #16    \n"
            "vsra.s32   q11, q1, #16    \n"
            "vsra.s32   q10, q2, #16    \n"
            "vsra.s32   q11, q3, #16    \n"

            "vshrn.s32  d20, q10, #2    \n"
            "vshrn.s32  d21, q11, #2    \n"
            "vqmovun.s16 d20, q10        \n"
            "vst1.8     {d20}, [%[dp]]!    \n"
            "subs       %[cnt], #1          \n"
            "bne        0b              \n"
            "sub        %[rows0p], #16         \n"
            "sub        %[rows1p], #16         \n"
            : [rows0p] "+r" (rows0p), [rows1p] "+r" (rows1p), [_b0] "+w" (_b0), [_b1] "+w" (_b1), \
                [cnt] "+r" (cnt), [dp] "+r" (dp_ptr)
            :
            : "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
        );

        //printf("resize_one_channel \n");
        }
#endif // __aarch64__
        for (; remain; --remain){
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *dp_ptr++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + \
                (short)((b1 * (short)(*rows1p++)) >> 16) + 2)>>2);
        }

        ibeta += 2;
    }

    delete[] buf;
    delete[] rowsbuf0;
    delete[] rowsbuf1;
}

void resize_one_channel_uv(const unsigned char* src, int w_in, int h_in, unsigned char* dst, int w_out, int h_out){

    const int resize_coef_bits = 11;
    const int resize_coef_scale = 1 << resize_coef_bits;

    double scale_x = (double)w_in / w_out;
    double scale_y = (double)h_in / h_out;

    int* buf = new int[w_out * 2 + h_out * 2];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w_out;//new int[h];

    short* ialpha = (short*)(buf + w_out + h_out);//new short[w * 2];
    short* ibeta = (short*)(buf + w_out * 2 + h_out);//new short[h * 2];

    float fx = 0.f;
    float fy = 0.f;
    int sx = 0.f;
    int sy = 0.f;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);
// #pragma omp parallel for
    for (int dx = 0; dx < w_out / 2; dx++){
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = floor(fx);
        fx -= sx;

        if (sx < 0){
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w_in - 1){
            sx = w_in - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        float a0 = (1.f - fx) * resize_coef_scale;
        float a1 =        fx  * resize_coef_scale;

        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }
// #pragma omp parallel for
    for (int dy = 0; dy < h_out; dy++) {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = floor(fy);
        fy -= sy;

        if (sy < 0){
            sy = 0;
            fy = 0.f;
        }
        if (sy >= h_in - 1){
            sy = h_in - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * resize_coef_scale;
        float b1 =        fy  * resize_coef_scale;

        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }
    // for (int dy = 0; dy < h_out; dy++) {
    //     printf("dy: %d, sy: %d \n", dy, yofs[dy]);
    // }

#undef SATURATE_CAST_SHORT
    // loop body
    short* rowsbuf0 = new short[w_out + 1];
    short* rowsbuf1 = new short[w_out + 1];
    short* rows0 = rowsbuf0;
    short* rows1 = rowsbuf1;

    int prev_sy1 = -1;
    //(x0 * a1 + x1 * a0) * b0 + (x2 * a1 + x3 * a0) * b1
    // for (int i = 0; i < w_out; i++)
    //     printf("%.2f ", ialpha[i] / 2048.0);
    // printf("\n");
    // for (int i = 0; i < h_out * 2; i++)
    //     printf("%.2f ", ibeta[i] / 2048.0);
    // printf("\n");
    // for (int i = 0; i < w_out / 2; i++)
    //     printf("%d ", xofs[i]);
    // printf("\n");
    // for (int i = 0; i < h_out; i++)
    //     printf("%d ", yofs[i]);
    // printf("\n");

   // printf("prev_sy1 : %d \n", prev_sy1);
    for (int dy = 0; dy < h_out; dy++){
        int sy = yofs[dy];
       // printf("dy, %d, sy: %d\n", dy, sy);
        if (sy == prev_sy1){
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + w_in * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w_out / 2; dx++){
                int sx = xofs[dx] * 2;
                short a0 = ialphap[0];
                short a1 = ialphap[1];
                const unsigned char* S1p = S1 + sx;
                int tmp = dx * 2;
                rows1p[tmp] = (S1p[0] * a0 + S1p[2] * a1) >> 4;
                rows1p[tmp + 1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;

                ialphap += 2;
            }
        }else{
            // hresize two rows
            const unsigned char *S0 = src + w_in * (sy);
            const unsigned char *S1 = src + w_in * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w_out / 2; dx++){
                int sx = xofs[dx] * 2;
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
                int tmp = dx * 2;
                rows0p[tmp] = (S0p[0] * a0 + S0p[2] * a1) >> 4;
                rows1p[tmp] = (S1p[0] * a0 + S1p[2] * a1) >> 4;

                rows0p[tmp + 1] = (S0p[1] * a0 + S0p[3] * a1) >> 4;
                rows1p[tmp + 1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;
                ialphap += 2;
            }
        }
        prev_sy1 = sy + 1;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* dp_ptr = dst + w_out * (dy);

        int cnt = w_out >> 3;
        int remain = w_out - (cnt << 3);
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);

#if 1// __aarch64__
        //printf("cnt : %d \n", cnt);
// #pragma omp parallel for
        for (cnt = w_out >> 3; cnt > 0; cnt--){
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16); // _acc >> 16 + _rows0p_sr4_mb0 >> 16
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2); // _acc >> 2
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _dout = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(dp_ptr, _dout);

            dp_ptr += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
#pragma omp parallel for
        if (cnt > 0){
        asm volatile(
            "mov        r4, #2          \n"
            "vdup.s32   q12, r4         \n"
             "0:                         \n"
            "pld        [%[rows0p], #128]      \n"
            "pld        [%[rows1p], #128]      \n"
            "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
            "vld1.s16   {d6-d7}, [%[rows0p]]!\n"
            "pld        [%[rows0p], #128]      \n"
            "pld        [%[rows1p], #128]      \n"
            "vmull.s16  q0, d2, %[_b0]     \n"
            "vmull.s16  q1, d3, %[_b0]     \n"
            "vmull.s16  q2, d6, %[_b1]     \n"
            "vmull.s16  q3, d7, %[_b1]     \n"

            "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
            "vld1.s16   {d6-d7}, [%[rows0p]]!\n"

            "vorr.s32   q10, q12, q12   \n"
            "vorr.s32   q11, q12, q12   \n"
            "vsra.s32   q10, q0, #16    \n"
            "vsra.s32   q11, q1, #16    \n"
            "vsra.s32   q10, q2, #16    \n"
            "vsra.s32   q11, q3, #16    \n"

            "vmull.s16  q0, d2, %[_b0]     \n"
            "vmull.s16  q1, d3, %[_b0]     \n"
            "vmull.s16  q2, d6, %[_b1]     \n"
            "vmull.s16  q3, d7, %[_b1]     \n"

            "vsra.s32   q10, q0, #16    \n"
            "vsra.s32   q11, q1, #16    \n"
            "vsra.s32   q10, q2, #16    \n"
            "vsra.s32   q11, q3, #16    \n"

            "vshrn.s32  d20, q10, #2    \n"
            "vshrn.s32  d21, q11, #2    \n"
            "vqmovun.s16 d20, q10        \n"
            "vst1.8     {d20}, [%[dp]]!    \n"
            "subs       %[cnt], #1          \n"
            "bne        0b              \n"
            "sub        %[rows0p], #16         \n"
            "sub        %[rows1p], #16         \n"
            : [rows0p] "+r" (rows0p), [rows1p] "+r" (rows1p), [_b0] "+w" (_b0), [_b1] "+w" (_b1), \
                [cnt] "+r" (cnt), [dp] "+r" (dp_ptr)
            :
            : "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
        );

        //printf("resize_one_channel \n");
        }
#endif // __aarch64__
        //printf("remain : %d \n", remain);
        for (; remain; --remain){
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *dp_ptr++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + \
                (short)((b1 * (short)(*rows1p++)) >> 16) + 2)>>2);
        }

        // dp_ptr = dst + w_out * (dy);
        // for (int i = 0; i < w_out; i++){
        //         printf("%d, ", dp_ptr[i]);
        //     }
        //     printf("\n");

        ibeta += 2;
    }

    delete[] buf;
    delete[] rowsbuf0;
    delete[] rowsbuf1;
}
