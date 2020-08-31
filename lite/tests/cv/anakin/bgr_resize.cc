#include <limits.h>
#include "lite/tests/cv/anakin/cv_utils.h"
void resize_three_channel(const uint8_t* src, int w_in, int h_in, uint8_t* dst, int w_out, int h_out);
void bgr_resize(const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out){
    LCHECK_GE(w_in, 0, "width must great than 0");
    LCHECK_GE(h_in, 0, "height must great than 0");
    LCHECK_GE(w_out, 0, "width must great than 0");
    LCHECK_GE(h_out, 0, "height must great than 0");
    if (w_out == w_in && h_out == h_in){
        memcpy(dst, src, sizeof(char) * w_in * h_in * 3);
        return;
    }
    //y
    resize_three_channel(src, w_in * 3, h_in, dst, w_out * 3, h_out);
}
void resize_three_channel(const uint8_t* src, int w_in, int h_in, uint8_t* dst, int w_out, int h_out){
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
    for (int dx = 0; dx < w_out / 3; dx++){
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
        xofs[dx] = sx * 3;
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
    for (int dy = 0; dy < h_out; dy++){
        int sy = yofs[dy];
       // printf("dy, %d, sy: %d\n", dy, sy);
        if (sy == prev_sy1){
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const uint8_t *S1 = src + w_in * (sy + 1);
            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w_out / 3; dx++){
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];
                const uint8_t* S1p = S1 + sx;
                int tmp = dx * 3;
                rows1p[tmp] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows1p[tmp + 1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
                rows1p[tmp + 2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;
                ialphap += 2;
            }
        }else{
            // hresize two rows
            const uint8_t *S0 = src + w_in * (sy);
            const uint8_t *S1 = src + w_in * (sy + 1);
            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w_out / 3; dx++){
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];
                const uint8_t* S0p = S0 + sx;
                const uint8_t* S1p = S1 + sx;
                int tmp = dx * 3;
                rows0p[tmp] = (S0p[0] * a0 + S0p[3] * a1) >> 4;
                rows1p[tmp] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows0p[tmp + 1] = (S0p[1] * a0 + S0p[4] * a1) >> 4;
                rows1p[tmp + 1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
                rows0p[tmp + 2] = (S0p[2] * a0 + S0p[5] * a1) >> 4;
                rows1p[tmp + 2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;
                ialphap += 2;
            }
        }
        prev_sy1 = sy + 1;
        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];
        short* rows0p = rows0;
        short* rows1p = rows1;
        uint8_t* dp_ptr = dst + w_out * (dy);
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
#endif // __aarch64__
        //printf("remain : %d \n", remain);
        for (; remain; --remain){
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *dp_ptr++ = (uint8_t)(((short)((b0 * (short)(*rows0p++)) >> 16) + \
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