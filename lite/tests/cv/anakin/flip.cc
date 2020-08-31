#include "lite/tests/cv/anakin/cv_utils.h"
void flip_x(const unsigned char* src, unsigned char* dst, int w_in, int h_in);

void flip_y(const unsigned char* src, unsigned char* dst, int w_in, int h_in);

void flip_xy(const unsigned char* src, unsigned char* dst, int w_in, int h_in);

//x: flip_num = 1 y: flip_num = -1 xy: flip_num = 0;
void flip(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int flip_num){

    if (flip_num == 1){//x
        flip_x(src, dst, w_in, h_in);
    }
    if (flip_num == -1){//y
        flip_y(src, dst, w_in, h_in);
    }
    if (flip_num == 0){//xy
        flip_xy(src, dst, w_in, h_in);
    }

}
/*
1 2 3
4 5 6
7 8 9
rotate:
7 8 9
4 5 6
1 2 3
*/
#ifdef __aarch64__
void flip_x(const unsigned char* src, unsigned char* dst, int w_in, int h_in){
    int h = h_in - 1;
    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < h_in; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;

        unsigned char* outptr0 = dst + (h - i) * w_in;//last
        unsigned char* outptr1 = outptr0 - w_in;
        unsigned char* outptr2 = outptr1 - w_in;
        unsigned char* outptr3 = outptr2 - w_in;

       // printf("outptr0: %x \n", outptr0);
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
        :
        :[ptr0] "r"(inptr0), [ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
        :"memory"
        );
        int j = 0;
        for (; j < w_in - 7; j += 8){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 3:
                        inptr0 = zerobuff;
                        outptr0 = zerobuff;
                    case 2:
                        inptr1 = zerobuff;
                        outptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                        outptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        outptr3 = zerobuff;
                    default:
                        break;
                }
            }
            // uint8x8_t va = vld1_u8(inptr0);
            // uint8x8_t a = vrev64_u8(va);
            // printf("va: %d, %d, %d, %d, %d, %d, %d, %d\n", va[0], va[1], va[2], va[3],va[4],va[5],va[6],va[7]);
            // printf("a: %d, %d, %d, %d, %d, %d, %d, %d \n", a[0], a[1], a[2],a[3],a[4],a[5],a[6],a[7]);
            asm volatile (
                "ld1  {v0.8b}, [%[inptr0]], #8    \n" //v0={00,01,02, 03, 04, 05, 06, 07}"
                "ld1  {v1.8b}, [%[inptr1]], #8     \n" //v0={10,11,12, 13, 14, 15, 16, 17}"
                "ld1  {v2.8b}, [%[inptr2]], #8    \n" //v0={20,21,22, 23, 24, 25, 26, 27}"
                "ld1  {v3.8b}, [%[inptr3]], #8    \n" //v0={30,31,32, 33, 34, 35, 36, 37}"

                "prfm   pldl1keep, [%[inptr0]]        \n"
                "prfm   pldl1keep, [%[inptr1]]        \n"
                "prfm   pldl1keep, [%[inptr2]]        \n"
                "prfm   pldl1keep, [%[inptr3]]        \n"

                "st1 {v0.8b}, [%[outptr0]], #8             \n" //00 10 20 30 04 14 24 34
                "st1 {v1.8b}, [%[outptr1]], #8              \n" //02 12 22 32
                "st1 {v2.8b}, [%[outptr2]], #8             \n" //01 11 21 31
                "st1 {v3.8b}, [%[outptr3]], #8              \n" //03 13 23 33

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            );
           // printf("outptr0: %x \n", outptr0);

           // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
        }
        for (; j < w_in; j++){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 0:
                        *outptr2++ = *inptr2++;
                    case 1:
                        *outptr1++ = *inptr1++;
                    case 2:
                        *outptr0++ = *inptr0++;
                    case 3:
                        //inptr3 = zerobuff;
                    default:
                        break;
                }
            }else{
                *outptr3++ = *inptr3++;
                *outptr2++ = *inptr2++;
                *outptr1++ = *inptr1++;
                *outptr0++ = *inptr0++;
            }
            // *outptr0-- = *inptr0++;
            // *outptr1-- = *inptr1++;
            // *outptr2-- = *inptr2++;
            // *outptr3-- = *inptr3++;
        }
    }
}
#else
void flip_x(const unsigned char* src, unsigned char* dst, int w_in, int h_in){
    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int h = h_in - 1;
    //4*8
    //printf("dst: %x \n", dst);
    for (int i = 0; i < h_in; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;

        unsigned char* outptr0 = dst + (h - i) * w_in;//last
        unsigned char* outptr1 = outptr0 - w_in;
        unsigned char* outptr2 = outptr1 - w_in;
        unsigned char* outptr3 = outptr2 - w_in;
        //printf("outptr0: %x \n", outptr0);
        asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
                "pld [%[ptr1]]            @ preload a, 64byte\n"
                "pld [%[ptr2]]            @ preload a, 64byte\n"
                "pld [%[ptr3]]            @ preload a, 64byte\n"
        :
        :[ptr0] "r"(inptr0), [ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
        :"memory"
        );
        int j = 0;
        for (; j < w_in - 7; j += 8){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 3:
                        inptr0 = zerobuff;
                        outptr0 = zerobuff;
                    case 2:
                        inptr1 = zerobuff;
                        outptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                        outptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        outptr3 = zerobuff;
                    default:
                        break;
                }
            }
            // uint8x8_t va = vld1_u8(inptr0);
            // uint8x8_t a = vrev64_u8(va);
            // printf("va: %d, %d, %d, %d, %d, %d, %d, %d\n", va[0], va[1], va[2], va[3],va[4],va[5],va[6],va[7]);
            // printf("a: %d, %d, %d, %d, %d, %d, %d, %d \n", a[0], a[1], a[2],a[3],a[4],a[5],a[6],a[7]);
            asm volatile (
                "vld1.8  {d0}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 04 05 06 07\n"
                "vld1.8  {d4}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 14 15 16 17\n"
                "vld1.8  {d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 24 25 26 27\n"
                "vld1.8  {d12}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 33 34 35 36 37\n"

                "pld [%[inptr0]]                         @ preload a, 64byte\n"
                "pld [%[inptr1]]                         @ preload a, 64byte\n"
                "pld [%[inptr2]]                         @ preload a, 64byte\n"
                "pld [%[inptr3]]                         @ preload a, 64byte\n"

                "vst1.32  {d0},    [%[outptr0]]!   @ write d0(q0,low),r00,r10 20 30\n"
                "vst1.32  {d4},    [%[outptr1]]!   @ write d4(q0,low),r01,r11 21 31\n"
                "vst1.32  {d8},    [%[outptr2]]!   @ write d4(q0,low),r01,r11 21 31\n"
                "vst1.32  {d12},    [%[outptr3]]!   @ write d4(q0,low),r01,r11 21 31\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
           // printf("outptr0: %x \n", outptr0);

           // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
        }
        for (; j < w_in; j++){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 0:
                        *outptr2++ = *inptr2++;
                    case 1:
                        *outptr1++ = *inptr1++;
                        //inptr1 = zerobuff;
                    case 2:
                        *outptr0++ = *inptr0++;
                    case 3:
                        //inptr3 = zerobuff;
                    default:
                        break;
                }
            }else{
                *outptr3++ = *inptr3++;
                *outptr2++ = *inptr2++;
                *outptr1++ = *inptr1++;
                *outptr0++ = *inptr0++;
            }
        }
    }
}
#endif
/*
1 2 3
4 5 6
7 8 9
flip:
3 2 1
6 5 4
9 8 7
*/
#ifdef __aarch64__
void flip_y(const unsigned char* src, unsigned char* dst, int w_in, int h_in){
    int stride_w = 8;
    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < h_in; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;

        unsigned char* outptr0 = dst + (i + 1) * w_in - stride_w;//last col
        unsigned char* outptr1 = outptr0 + w_in;
        unsigned char* outptr2 = outptr1 + w_in;
        unsigned char* outptr3 = outptr2 + w_in;

       // printf("outptr0: %x \n", outptr0);
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
        :
        :[ptr0] "r"(inptr0), [ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
        :"memory"
        );
        int j = 0;
        for (; j < w_in - 7; j += 8){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 3:
                        inptr0 = zerobuff;
                        outptr0 = zerobuff;
                    case 2:
                        inptr1 = zerobuff;
                        outptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                        outptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        outptr3 = zerobuff;
                    default:
                        break;
                }
            }
            // uint8x8_t va = vld1_u8(inptr0);
            // uint8x8_t a = vrev64_u8(va);
            // printf("va: %d, %d, %d, %d, %d, %d, %d, %d\n", va[0], va[1], va[2], va[3],va[4],va[5],va[6],va[7]);
            // printf("a: %d, %d, %d, %d, %d, %d, %d, %d \n", a[0], a[1], a[2],a[3],a[4],a[5],a[6],a[7]);
            asm volatile (
                "ld1  {v0.8b}, [%[inptr0]], #8    \n" //v0={00,01,02, 03, 04, 05, 06, 07}"
                "ld1  {v1.8b}, [%[inptr1]], #8     \n" //v0={10,11,12, 13, 14, 15, 16, 17}"
                "ld1  {v2.8b}, [%[inptr2]], #8    \n" //v0={20,21,22, 23, 24, 25, 26, 27}"
                "ld1  {v3.8b}, [%[inptr3]], #8    \n" //v0={30,31,32, 33, 34, 35, 36, 37}"

                "rev64  v4.8b, v0.8b                \n" //@ reverse 07 06 05 04 03 02 01 00
                "rev64  v5.8b, v1.8b                \n" //@ reverse 07 06 05 04 03 02 01 00
                "rev64  v6.8b, v2.8b                \n" //@ reverse 07 06 05 04 03 02 01 00
                "rev64  v7.8b, v3.8b                \n" //@ reverse 07 06 05 04 03 02 01 00

                "prfm   pldl1keep, [%[inptr0]]        \n"
                "prfm   pldl1keep, [%[inptr1]]        \n"
                "prfm   pldl1keep, [%[inptr2]]        \n"
                "prfm   pldl1keep, [%[inptr3]]        \n"

                "st1 {v4.8b}, [%[outptr0]]             \n" //00 10 20 30 04 14 24 34
                "st1 {v5.8b}, [%[outptr1]]              \n" //02 12 22 32
                "st1 {v6.8b}, [%[outptr2]]             \n" //01 11 21 31
                "st1 {v7.8b}, [%[outptr3]]              \n" //03 13 23 33

                "sub %[outptr0], %[outptr0], %[stride_w]       \n" //@ ptr - stride_w
                "sub %[outptr1], %[outptr1], %[stride_w]       \n"
                "sub %[outptr2], %[outptr2], %[stride_w]       \n"
                "sub %[outptr3], %[outptr3], %[stride_w]       \n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), \
              [stride_w] "+r" (stride_w)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            );
           // printf("outptr0: %x \n", outptr0);

           // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
        }
        outptr3 += stride_w - 1;
        outptr2 += stride_w - 1;
        outptr1 += stride_w - 1;
        outptr0 += stride_w - 1;
        for (; j < w_in; j++){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 0:
                        *outptr2-- = *inptr2++;
                    case 1:
                        *outptr1-- = *inptr1++;
                        //inptr1 = zerobuff;
                    case 2:
                        *outptr0-- = *inptr0++;
                    case 3:
                        //inptr3 = zerobuff;
                    default:
                        break;
                }
            }else{
                *outptr3-- = *inptr3++;
                *outptr2-- = *inptr2++;
                *outptr1-- = *inptr1++;
                *outptr0-- = *inptr0++;
            }
            // *outptr0-- = *inptr0++;
            // *outptr1-- = *inptr1++;
            // *outptr2-- = *inptr2++;
            // *outptr3-- = *inptr3++;
        }
    }
}
#else
void flip_y(const unsigned char* src, unsigned char* dst, int w_in, int h_in){
    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int stride_w = 8;
    //4*8
    //printf("dst: %x \n", dst);
    for (int i = 0; i < h_in; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;

        unsigned char* outptr0 = dst + (i + 1) * w_in - stride_w;//last
        unsigned char* outptr1 = outptr0 + w_in;
        unsigned char* outptr2 = outptr1 + w_in;
        unsigned char* outptr3 = outptr2 + w_in;
        //printf("outptr0: %x \n", outptr0);
        asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
                "pld [%[ptr1]]            @ preload a, 64byte\n"
                "pld [%[ptr2]]            @ preload a, 64byte\n"
                "pld [%[ptr3]]            @ preload a, 64byte\n"
        :
        :[ptr0] "r"(inptr0), [ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
        :"memory"
        );
        int j = 0;
        for (; j < w_in - 7; j += 8){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 3:
                        inptr0 = zerobuff;
                        outptr0 = zerobuff;
                    case 2:
                        inptr1 = zerobuff;
                        outptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                        outptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        outptr3 = zerobuff;
                    default:
                        break;
                }
            }
            // uint8x8_t va = vld1_u8(inptr0);
            // uint8x8_t a = vrev64_u8(va);
            // printf("va: %d, %d, %d, %d, %d, %d, %d, %d\n", va[0], va[1], va[2], va[3],va[4],va[5],va[6],va[7]);
            // printf("a: %d, %d, %d, %d, %d, %d, %d, %d \n", a[0], a[1], a[2],a[3],a[4],a[5],a[6],a[7]);
            asm volatile (
                "vld1.8  {d0}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 04 05 06 07\n"
                "vld1.8  {d4}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 14 15 16 17\n"
                "vld1.8  {d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 24 25 26 27\n"
                "vld1.8  {d12}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 33 34 35 36 37\n"

                "vrev64.8  d1, d0               @ reverse 07 06 05 04 03 02 01 00 \n"
                "vrev64.8  d5, d4              @ reverse 07 06 05 04 03 02 01 00 \n"
                "vrev64.8  d9, d8               @ reverse 07 06 05 04 03 02 01 00 \n"
                "vrev64.8  d13, d12               @ reverse 07 06 05 04 03 02 01 00 \n"

                "pld [%[inptr0]]                         @ preload a, 64byte\n"
                "pld [%[inptr1]]                         @ preload a, 64byte\n"
                "pld [%[inptr2]]                         @ preload a, 64byte\n"
                "pld [%[inptr3]]                         @ preload a, 64byte\n"

                "vst1.32  {d1},    [%[outptr0]]   @ write d0(q0,low),r00,r10 20 30\n"
                "vst1.32  {d5},    [%[outptr1]]   @ write d4(q0,low),r01,r11 21 31\n"
                "vst1.32  {d9},    [%[outptr2]]   @ write d4(q0,low),r01,r11 21 31\n"
                "vst1.32  {d13},    [%[outptr3]]   @ write d4(q0,low),r01,r11 21 31\n"

                "sub %[outptr0], %[stride_w]       @ ptr - stride_w \n"
                "sub %[outptr1], %[stride_w]       @ ptr - stride_w \n"
                "sub %[outptr2], %[stride_w]       @ ptr - stride_w \n"
                "sub %[outptr3], %[stride_w]       @ ptr - stride_w \n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), \
              [stride_w] "+r" (stride_w)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
           // printf("outptr0: %x \n", outptr0);

           // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
        }
        outptr3 += stride_w - 1;
        outptr2 += stride_w - 1;
        outptr1 += stride_w - 1;
        outptr0 += stride_w - 1;
        for (; j < w_in; j++){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 0:
                        *outptr2-- = *inptr2++;
                    case 1:
                        *outptr1-- = *inptr1++;
                        //inptr1 = zerobuff;
                    case 2:
                        *outptr0-- = *inptr0++;
                    case 3:
                        //inptr3 = zerobuff;
                    default:
                        break;
                }
            }else{
                *outptr3-- = *inptr3++;
                *outptr2-- = *inptr2++;
                *outptr1-- = *inptr1++;
                *outptr0-- = *inptr0++;
            }
            // *outptr0-- = *inptr0++;
            // *outptr1-- = *inptr1++;
            // *outptr2-- = *inptr2++;
            // *outptr3-- = *inptr3++;
        }
    }
}
#endif
/*
1 2 3
4 5 6
7 8 9
flip:
9 8 7
6 5 4
3 2 1
*/
#ifdef __aarch64__
void flip_xy(const unsigned char* src, unsigned char* dst, int w_in, int h_in){
    int stride_w = 8;
    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < h_in; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;

        unsigned char* outptr0 = dst + (h_in - i) * w_in - stride_w;//last col
        unsigned char* outptr1 = outptr0 - w_in;
        unsigned char* outptr2 = outptr1 - w_in;
        unsigned char* outptr3 = outptr2 - w_in;

       // printf("outptr0: %x \n", outptr0);
        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
        :
        :[ptr0] "r" (inptr0),[ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
        :"memory"
        );
        int j = 0;
        for (; j < w_in - 7; j += 8){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 3:
                        inptr0 = zerobuff;
                        outptr0 = zerobuff;
                    case 2:
                        inptr1 = zerobuff;
                        outptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                        outptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        outptr3 = zerobuff;
                    default:
                        break;
                }
            }
            // uint8x8_t va = vld1_u8(inptr0);
            // uint8x8_t a = vrev64_u8(va);
            // printf("va: %d, %d, %d, %d, %d, %d, %d, %d\n", va[0], va[1], va[2], va[3],va[4],va[5],va[6],va[7]);
            // printf("a: %d, %d, %d, %d, %d, %d, %d, %d \n", a[0], a[1], a[2],a[3],a[4],a[5],a[6],a[7]);
            asm volatile (
                "ld1  {v0.8b}, [%[inptr0]], #8    \n" //v0={00,01,02, 03, 04, 05, 06, 07}"
                "ld1  {v1.8b}, [%[inptr1]], #8     \n" //v0={10,11,12, 13, 14, 15, 16, 17}"
                "ld1  {v2.8b}, [%[inptr2]], #8    \n" //v0={20,21,22, 23, 24, 25, 26, 27}"
                "ld1  {v3.8b}, [%[inptr3]], #8    \n" //v0={30,31,32, 33, 34, 35, 36, 37}"

                "rev64  v4.8b, v0.8b                \n" //@ reverse 07 06 05 04 03 02 01 00
                "rev64  v5.8b, v1.8b                \n" //@ reverse 07 06 05 04 03 02 01 00
                "rev64  v6.8b, v2.8b                \n" //@ reverse 07 06 05 04 03 02 01 00
                "rev64  v7.8b, v3.8b                \n" //@ reverse 07 06 05 04 03 02 01 00

                "prfm   pldl1keep, [%[inptr0]]        \n"
                "prfm   pldl1keep, [%[inptr1]]        \n"
                "prfm   pldl1keep, [%[inptr2]]        \n"
                "prfm   pldl1keep, [%[inptr3]]        \n"

                "st1 {v4.8b}, [%[outptr0]]             \n" //00 10 20 30 04 14 24 34
                "st1 {v5.8b}, [%[outptr1]]              \n" //02 12 22 32
                "st1 {v6.8b}, [%[outptr2]]             \n" //01 11 21 31
                "st1 {v7.8b}, [%[outptr3]]              \n" //03 13 23 33

                "sub %[outptr0], %[outptr0], %[stride_w]       \n" //@ ptr - stride_w
                "sub %[outptr1], %[outptr1], %[stride_w]       \n"
                "sub %[outptr2], %[outptr2], %[stride_w]       \n"
                "sub %[outptr3], %[outptr3], %[stride_w]       \n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), \
              [stride_w] "+r" (stride_w)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            );
           // printf("outptr0: %x \n", outptr0);

           // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
        }
        outptr3 += stride_w - 1;
        outptr2 += stride_w - 1;
        outptr1 += stride_w - 1;
        outptr0 += stride_w - 1;
        for (; j < w_in; j++){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 0:
                        *outptr2-- = *inptr2++;
                    case 1:
                        *outptr1-- = *inptr1++;
                        //inptr1 = zerobuff;
                    case 2:
                        *outptr0-- = *inptr0++;
                    case 3:
                        //inptr3 = zerobuff;
                    default:
                        break;
                }
            }else{
                *outptr3-- = *inptr3++;
                *outptr2-- = *inptr2++;
                *outptr1-- = *inptr1++;
                *outptr0-- = *inptr0++;
            }
            // *outptr0-- = *inptr0++;
            // *outptr1-- = *inptr1++;
            // *outptr2-- = *inptr2++;
            // *outptr3-- = *inptr3++;
        }
    }
}
#else
void flip_xy(const unsigned char* src, unsigned char* dst, int w_in, int h_in){
    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int stride_w = 8;
    //4*8
    //printf("dst: %x \n", dst);
    for (int i = 0; i < h_in; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;

        unsigned char* outptr0 = dst + (h_in - i) * w_in - stride_w;//last
        unsigned char* outptr1 = outptr0 - w_in;
        unsigned char* outptr2 = outptr1 - w_in;
        unsigned char* outptr3 = outptr2 - w_in;
        //printf("outptr0: %x \n", outptr0);
        asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
                "pld [%[ptr1]]            @ preload a, 64byte\n"
                "pld [%[ptr2]]            @ preload a, 64byte\n"
                "pld [%[ptr3]]            @ preload a, 64byte\n"
        :
        :[ptr0] "r"(inptr0), [ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
        :"memory"
        );
        int j = 0;
        for (; j < w_in - 7; j += 8){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 3:
                        inptr0 = zerobuff;
                        outptr0 = zerobuff;
                    case 2:
                        inptr1 = zerobuff;
                        outptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                        outptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        outptr3 = zerobuff;
                    default:
                        break;
                }
            }
            // uint8x8_t va = vld1_u8(inptr0);
            // uint8x8_t a = vrev64_u8(va);
            // printf("va: %d, %d, %d, %d, %d, %d, %d, %d\n", va[0], va[1], va[2], va[3],va[4],va[5],va[6],va[7]);
            // printf("a: %d, %d, %d, %d, %d, %d, %d, %d \n", a[0], a[1], a[2],a[3],a[4],a[5],a[6],a[7]);
            asm volatile (
                "vld1.8  {d0}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 04 05 06 07\n"
                "vld1.8  {d4}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 14 15 16 17\n"
                "vld1.8  {d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 24 25 26 27\n"
                "vld1.8  {d12}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 33 34 35 36 37\n"

                "vrev64.8  d1, d0               @ reverse 07 06 05 04 03 02 01 00 \n"
                "vrev64.8  d5, d4              @ reverse 07 06 05 04 03 02 01 00 \n"
                "vrev64.8  d9, d8               @ reverse 07 06 05 04 03 02 01 00 \n"
                "vrev64.8  d13, d12               @ reverse 07 06 05 04 03 02 01 00 \n"

                "pld [%[inptr0]]                         @ preload a, 64byte\n"
                "pld [%[inptr1]]                         @ preload a, 64byte\n"
                "pld [%[inptr2]]                         @ preload a, 64byte\n"
                "pld [%[inptr3]]                         @ preload a, 64byte\n"

                "vst1.32  {d1},    [%[outptr0]]   @ write d0(q0,low),r00,r10 20 30\n"
                "vst1.32  {d5},    [%[outptr1]]   @ write d4(q0,low),r01,r11 21 31\n"
                "vst1.32  {d9},    [%[outptr2]]   @ write d4(q0,low),r01,r11 21 31\n"
                "vst1.32  {d13},    [%[outptr3]]   @ write d4(q0,low),r01,r11 21 31\n"

                "sub %[outptr0], %[stride_w]       @ ptr - stride_w \n"
                "sub %[outptr1], %[stride_w]       @ ptr - stride_w \n"
                "sub %[outptr2], %[stride_w]       @ ptr - stride_w \n"
                "sub %[outptr3], %[stride_w]       @ ptr - stride_w \n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), \
              [stride_w] "+r" (stride_w)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
           // printf("outptr0: %x \n", outptr0);

           // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
        }
        outptr3 += stride_w - 1;
        outptr2 += stride_w - 1;
        outptr1 += stride_w - 1;
        outptr0 += stride_w - 1;
        for (; j < w_in; j++){
            if (i + 3 >= h_in){
                switch ((i + 3) - h_in){
                    case 0:
                        *outptr2-- = *inptr2++;
                    case 1:
                        *outptr1-- = *inptr1++;
                        //inptr1 = zerobuff;
                    case 2:
                        *outptr0-- = *inptr0++;
                    case 3:
                        //inptr3 = zerobuff;
                    default:
                        break;
                }
            }else{
                *outptr3-- = *inptr3++;
                *outptr2-- = *inptr2++;
                *outptr1-- = *inptr1++;
                *outptr0-- = *inptr0++;
            }
            // *outptr0-- = *inptr0++;
            // *outptr1-- = *inptr1++;
            // *outptr2-- = *inptr2++;
            // *outptr3-- = *inptr3++;
        }
    }
}
#endif
