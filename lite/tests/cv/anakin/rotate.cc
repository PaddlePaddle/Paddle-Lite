#include "lite/tests/cv/anakin/cv_utils.h"
void rotate90(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out);

void rotate270(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out);

void rotate180(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out);

void rotate(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int angle){

    if (angle == 90){
        rotate90(src, dst, w_in, h_in, h_in, w_in);
    }
    if (angle == 270){
        rotate270(src, dst, w_in, h_in, h_in, w_in);
    }
    if (angle == 180){
        rotate180(src, dst, w_in, h_in, w_in, h_in);
    }

}

/*
1 2 3
4 5 6
7 8 9
rotate:
1 4 7
2 5 8
3 6 9
*/
//transpose
#ifdef __aarch64__
void rotate90(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out){

    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int stride_h = 4 * w_in;
    int stride_h_w = 4 * w_in - 8;
    //block 8*8. -- 8*8
    int i = 0;
    for (i = 0; i < h_in - 7; i += 8){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;
        // const unsigned char* inptr4 = inptr3 + w_in;
        // const unsigned char* inptr5 = inptr4 + w_in;
        // const unsigned char* inptr6 = inptr5 + w_in;
        // const unsigned char* inptr7 = inptr6 + w_in;

        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr0], #64]   \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr1], #64]   \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr2], #64]   \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
                "prfm   pldl1keep, [%[ptr3], #64]   \n"
                // "prfm   pldl1keep, [%[ptr4]]        \n"
                // "prfm   pldl1keep, [%[ptr4], #64]   \n"
                // "prfm   pldl1keep, [%[ptr5]]        \n"
                // "prfm   pldl1keep, [%[ptr5], #64]   \n"
        :
        :[ptr0] "r"(inptr0), [ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
        :"memory"
        );
        int j = 0;
        for (; j < w_in - 7; j += 8){
            unsigned char* outptr0 = dst + j * w_out + i;
            unsigned char* outptr1 = outptr0 + w_out;
            unsigned char* outptr2 = outptr1 + w_out;
            unsigned char* outptr3 = outptr2 + w_out;
            unsigned char* outptr4 = outptr3 + w_out;
            unsigned char* outptr5 = outptr4 + w_out;
            unsigned char* outptr6 = outptr5 + w_out;
            unsigned char* outptr7 = outptr6 + w_out;
            // printf("outptr0: %x, inptr0: %x \n", outptr0, inptr0);
            // printf("inptr0: %d, %d, %d, %d \n", inptr0[0], inptr0[1], inptr0[2], inptr0[3]);
            asm volatile (
                "ld1  {v0.8b}, [%[inptr0]]    \n" //v0={00,01,02, 03, 04, 05, 06, 07}"
                "ld1  {v1.8b}, [%[inptr1]]    \n" //v0={10,11,12, 13, 14, 15, 16, 17}"
                "ld1  {v2.8b}, [%[inptr2]]    \n" //v0={20,21,22, 23, 24, 25, 26, 27}"
                "ld1  {v3.8b}, [%[inptr3]]    \n" //v0={30,31,32, 33, 34, 35, 36, 37}"

                "add %[inptr0], %[inptr0], %[stride_h] \n" //4 +4*w_in
                "add %[inptr1], %[inptr1], %[stride_h] \n" //5
                "add %[inptr2], %[inptr2], %[stride_h] \n" //6
                "add %[inptr3], %[inptr3], %[stride_h] \n" //7

                "trn1 v4.8b, v0.8b, v1.8b             \n" //v4={00 10 02 12 04 14 06 16 }
                "trn1 v6.8b, v2.8b, v3.8b             \n" //v4={20 30 22 32 24 34 26 36 }

                "trn2 v5.8b, v0.8b, v1.8b             \n" //v5={01 11 03 13 05 15 07 17 }
                "trn2 v7.8b, v2.8b, v3.8b             \n" //v7={21 31 23 33 25 35 27 37 }

                "ld1  {v12.8b}, [%[inptr0]]    \n" //v0={00,01,02, 03, 04, 05, 06, 07}"
                "ld1  {v13.8b}, [%[inptr1]]    \n" //v0={10,11,12, 13, 14, 15, 16, 17}"
                "ld1  {v14.8b}, [%[inptr2]]    \n" //v0={20,21,22, 23, 24, 25, 26, 27}"
                "ld1  {v15.8b}, [%[inptr3]]    \n" //v0={30,31,32, 33, 34, 35, 36, 37}"

                "trn1 v0.4h, v4.4h, v6.4h             \n" //v0={00 10 20 30 04 14 24 34}
                "trn1 v2.4h, v5.4h, v7.4h             \n" //v2={01 11 21 31 05 15 25 35}

                "trn2 v1.4h, v4.4h, v6.4h             \n" //v1={02 12 22 32 06 16 26 36}
                "trn2 v3.4h, v5.4h, v7.4h             \n" //v3={03 13 23 33 07 17 27 37}

                "trn1 v9.8b, v12.8b, v13.8b             \n" //v9={40 50 42 52 44 54 46 56 }
                "trn1 v11.8b, v14.8b, v15.8b             \n" //v11={60 70 62 72 64 74 66 76 }

                "trn2 v10.8b, v12.8b, v13.8b             \n" //v10={01 11 03 13 05 15 07 17 }
                "trn2 v12.8b, v14.8b, v15.8b             \n" //v12={21 31 23 33 25 35 27 37 }

                "sub %[inptr0], %[inptr0], %[stride_h_w] \n" //4 - 4*w_in + 8
                "sub %[inptr1], %[inptr1], %[stride_h_w] \n" //5
                "sub %[inptr2], %[inptr2], %[stride_h_w] \n" //6
                "sub %[inptr3], %[inptr3], %[stride_h_w] \n" //7

                "trn1 v4.4h, v9.4h, v11.4h             \n" //v4={40 50 60 70 44 54 64 74}
                "trn1 v6.4h, v10.4h, v12.4h             \n" //v6={41 51 61 71 45 55 65 75}

                "trn2 v5.4h, v9.4h, v11.4h             \n" //v5={42 52 62 72 46 56 66 76}
                "trn2 v7.4h, v10.4h, v12.4h            \n" //v7={43 53 63 73 47 57 67 77}

                "trn1 v8.2s, v0.2s, v4.2s             \n" //v8={00 10 20 30 40 50 60 70}
                "trn1 v9.2s, v2.2s, v6.2s             \n" //v6={01 11 21 31 41 51 61 71}
                "trn1 v10.2s, v1.2s, v5.2s             \n" //v10={02 12 22 32 42 52 62 72}
                "trn1 v11.2s, v3.2s, v7.2s             \n" //v11={03 13 23 33 43 53 63 73}

                "trn2 v12.2s, v0.2s, v4.2s             \n" //v12={04 14 24 34 44 54 64 74}
                "trn2 v13.2s, v2.2s, v6.2s             \n" //v13={05 15 25 35 45 55 65 75}
                "trn2 v14.2s, v1.2s, v5.2s             \n" //v14={06 16 26 36 46 56 66 76}
                "trn2 v15.2s, v3.2s, v7.2s             \n" //v15={07 17 27 37 47 57 67 77}

                "st1 {v8.8b}, [%[outptr0]], #8             \n" //00 10 20 30 04 14 24 34
                "st1 {v9.8b}, [%[outptr1]], #8              \n" //02 12 22 32
                "st1 {v10.8b}, [%[outptr2]], #8              \n" //01 11 21 31
                "st1 {v11.8b}, [%[outptr3]], #8              \n" //03 13 23 33

                "st1 {v12.8b}, [%[outptr4]], #8             \n" //00 10 20 30 04 14 24 34
                "st1 {v13.8b}, [%[outptr5]], #8              \n" //02 12 22 32
                "st1 {v14.8b}, [%[outptr6]], #8              \n" //01 11 21 31
                "st1 {v15.8b}, [%[outptr7]], #8              \n" //03 13 23 33

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), \
              [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5), [outptr6] "+r"(outptr6), [outptr7] "+r"(outptr7), \
              [stride_h] "+r"(stride_h), [stride_h_w] "+r"(stride_h_w)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",  "v12", "v13", "v14", "v15"
            );
            //  printf("outptr0: %x, inptr0: %x \n", outptr0, inptr0);
            // // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
            // outptr0 -= 8;
            // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
            // printf("outptr0: %d, %d, %d, %d \n", outptr0[4], outptr0[5], outptr0[6], outptr0[7]);
        }
        const unsigned char* inptr4 = inptr3 + w_in;
        const unsigned char* inptr5 = inptr4 + w_in;
        const unsigned char* inptr6 = inptr5 + w_in;
        const unsigned char* inptr7 = inptr6 + w_in;
        for (; j < w_in; j++){
            unsigned char* outptr = dst + j * w_out + i;
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
            *outptr++ = *inptr6++;
            *outptr++ = *inptr7++;
        }

    }
    // if (i + 3 < h_in){//4
    //     //4*8
    //     const unsigned char* inptr0 = src + i * w_in;
    //     const unsigned char* inptr1 = inptr0 + w_in;
    //     const unsigned char* inptr2 = inptr1 + w_in;
    //     const unsigned char* inptr3 = inptr2 + w_in;
    //     asm volatile(
    //     "prfm   pldl1keep, [%[ptr0]]                \n"
    //             "prfm   pldl1keep, [%[ptr0], #64]   \n"
    //             "prfm   pldl1keep, [%[ptr1]]        \n"
    //             "prfm   pldl1keep, [%[ptr1], #64]   \n"
    //             "prfm   pldl1keep, [%[ptr2]]        \n"
    //             "prfm   pldl1keep, [%[ptr2], #64]   \n"
    //             "prfm   pldl1keep, [%[ptr3]]        \n"
    //             "prfm   pldl1keep, [%[ptr3], #64]   \n"
    //     :
    //     :[ptr0] "r"(inptr0),[ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
    //     :"memory"
    //     );
    //     int j = 0;
    //     for (; j < w_in - 7; j += 8){
    //         unsigned char* outptr0 = dst + j * w_out + i;
    //         unsigned char* outptr1 = outptr0 + w_out;
    //         unsigned char* outptr2 = outptr1 + w_out;
    //         unsigned char* outptr3 = outptr2 + w_out;
    //         unsigned char* outptr4 = outptr3 + w_out;
    //         unsigned char* outptr5 = outptr4 + w_out;
    //         unsigned char* outptr6 = outptr5 + w_out;
    //         unsigned char* outptr7 = outptr6 + w_out;

    //         uint8x8_t din0 = vld1_u8(inptr0); //00 01 02 03 04 05 06 07
    //         uint8x8_t din1 = vld1_u8(inptr1); //10 11 12 13 14 15 16 17
    //         uint8x8_t din2 = vld1_u8(inptr2); //20 21 22 23 24 25 26 27
    //         uint8x8_t din3 = vld1_u8(inptr3); //30 31 32 33 34 35 36 37

    //         uint8x8x2_t din0_1 = vtrn_u8(din0, din1); //00 10 02 12 04 14 06 16   01 11 03 13 05 15 07 17
    //         uint8x8x2_t din2_3 = vtrn_u8(din2, din3); //20 30 22 32 24 34 06 16   01 11 03 13 05 15 07 17

    //         uint16x8_t va = vmovl_u8(din0_1.val[0]); //00 10 02 12 04 14 06 16
    //         uint16x8_t vb = vmovl_u8(din0_1.val[1]); //01 11 03 13 05 15 07 17
    //         uint16x8_t vc = vmovl_u8(din0_1.val[0]); //20 30 22 32 24 34 26 36
    //         uint16x8_t vd = vmovl_u8(din0_1.val[1]); //21 31 23 33 25 35 27 37

    //         uint16x8x2_t vdata1 = vtrnq_u16(va, vc); //00 10 20 30 04 14 24 34    02 12 22 32 06 16 26 36
    //         uint16x8x2_t vdata2 = vtrnq_u16(vb, vd); //01 11 21 31 01 15 25 35    03 13 23 33 07 17 27 37

    //         uint8x8_t vout0 = vqmovn_u16(vdata1.val[0]); //00 10 20 30 04 14 24 34
    //         uint8x8_t vout1 = vqmovn_u16(vdata1.val[1]); //02 12 22 32 06 16 26 36
    //         uint8x8_t vout2 = vqmovn_u16(vdata2.val[0]); //01 11 21 31 01 15 25 35
    //         uint8x8_t vout3 = vqmovn_u16(vdata2.val[1]); //03 13 23 33 07 17 27 37

    //         for (int k = 0; k < 4; k++){
    //             *outptr0++ = vout0[0 + k];
    //             *outptr4++ = vout0[4 + k];
    //             *outptr2++ = vout1[0 + k];
    //             *outptr6++ = vout1[4 + k];
    //             *outptr1++ = vout2[0 + k];
    //             *outptr5++ = vout2[4 + k];
    //             *outptr3++ = vout3[0 + k];
    //             *outptr7++ = vout3[4 + k];
    //         }
    //         inptr0 += 8;
    //         inptr1 += 8;
    //         inptr2 += 8;
    //         inptr3 += 8;
    //     }

    //     for (; j < w_in; j++){
    //         unsigned char* outptr = dst + j * w_out + i;
    //         *outptr++ = *inptr0++;
    //         *outptr++ = *inptr1++;
    //         *outptr++ = *inptr2++;
    //         *outptr++ = *inptr3++;
    //     }
    //     i += 4;
    // }
    for (; i < h_in; i++){
        const unsigned char* inptr0 = src + i * w_in;
        for (int j = 0 ; j < w_in; j++){
            unsigned char* outptr0 = dst + j * w_out + i;
            *outptr0 = *inptr0++;
        }
    }

}
#else
void rotate90(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out){

    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int hremain = h_in % 4;
    //block 4*8. -- 8*4
    int i = 0;
    for (i = 0; i < h_in - 3; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;
        // const unsigned char* inptr4 = inptr3 + w_in;
        // const unsigned char* inptr5 = inptr4 + w_in;
        asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
                "pld [%[ptr0], #64]            @ preload a, 64byte\n"
                "pld [%[ptr1]]            @ preload a, 64byte\n"
                "pld [%[ptr1], #64]            @ preload a, 64byte\n"
                "pld [%[ptr2]]            @ preload a, 64byte\n"
                "pld [%[ptr2], #64]            @ preload a, 64byte\n"
                "pld [%[ptr3]]            @ preload a, 64byte\n"
                "pld [%[ptr3], #64]            @ preload a, 64byte\n"
                // "pld [%[ptr4]]            @ preload a, 64byte\n"
                // "pld [%[ptr4], #64]            @ preload a, 64byte\n"
                // "pld [%[ptr5]]            @ preload a, 64byte\n"
                // "pld [%[ptr5], #64]            @ preload a, 64byte\n"
        :
        :[ptr0] "r"(inptr0), [ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
       // [ptr4] "r"(inptr4),[ptr5] "r"(inptr5)
        :"memory"
        );
        int j = 0;
        for (j = 0; j < w_in - 7; j += 8){
           // printf("j: %d, inptr0: %x \n", j, inptr0);
            unsigned char* outptr0 = dst + j * w_out + i;
            unsigned char* outptr1 = outptr0 + w_out;
            unsigned char* outptr2 = outptr1 + w_out;
            unsigned char* outptr3 = outptr2 + w_out;
            unsigned char* outptr4 = outptr3 + w_out;
            unsigned char* outptr5 = outptr4 + w_out;
            unsigned char* outptr6 = outptr5 + w_out;
            unsigned char* outptr7 = outptr6 + w_out;
            asm volatile (
                "vld1.8  {d0}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 04 05 06 07\n"
                "vld1.8  {d4}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 14 15 16 17\n"
                "vld1.8  {d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 24 25 26 27\n"
                "vld1.8  {d12}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 33 34 35 36 37\n"

                "vtrn.8  d0, d4                  @ trans data: d0=r00,r10,r02,r12 04 14 06 16  d4=01 11 03 13 \n"
                "vtrn.8  d8, d12                  @ trans data: d8=r20,r30,r12,r32 24 34 26 36  d12=21 31 23 33 \n"
                // d0=r00,r10,r20,r30 04 14 24 34 d8 = 02 12 22 32 06 16 26 36
                "vtrn.16 d0, d8                  @ trans data: \n"
                // d4=01 11 21 31 05 15 25 35; d12 =r03 13 23 33 07 17 27 37
                "vtrn.16  d4, d12                  @ trans data:\n"
                //d0=00,10,20,30 01 11 21 31 d4 = 04 14 24 34 05 15 25 35
                "vtrn.32 d0, d4                  @ trans data: \n"
                //d8= 02 12 22 32 03 13 23 33; d12 =r06 16 26 36 07 17 27 37
                "vtrn.32  d8, d12                  @ trans data: \n"
                // "vtrn.16  d1, d9                   @ trans data: d1=r04,r14,r24,r34  d9=06 16 26 36 \n"

                "vst1.32  {d0[0]},    [%[outptr0]]!   @ write d0(q0,low),r00,r10 20 30\n"
                "vst1.32  {d0[1]},    [%[outptr1]]!   @ write d4(q0,low),r01,r11 21 31\n"
                "vst1.32  {d4[0]},    [%[outptr4]]!   @ write d2(q0,low),r02,r12\n"
                "vst1.32  {d4[1]},    [%[outptr5]]!   @ write d6(q0,low),r06,r16\n"
                "vst1.32  {d8[0]},    [%[outptr2]]!   @ write d1(q0,high),r02,r12 22 32\n"
                "vst1.32  {d8[1]},    [%[outptr3]]!   @ write d5(q0,low),r03,r13 23 33\n"
                "vst1.32 {d12[0]},     [%[outptr6]]!   @ write d3(q0,low),r03,r13\n"
                "vst1.32  {d12[1]},    [%[outptr7]]!   @ write d7(q0,low),r07,r17\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), \
              [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5), [outptr6] "+r"(outptr6), [outptr7] "+r"(outptr7)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
           //  printf("j: %d, outptr0: %x \n", j, outptr0);
        }
        //printf("j: %d, inptr0: %x\n", j, inptr0);
        for (; j < w_in; j++){
            unsigned char* outptr = dst + j * w_out + i;
           // printf("j: %d, outptr: %x \n", j, outptr);
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
        }

    }
    if (hremain > 0){
        for (; i < h_in; i++){
            const unsigned char* inptr0 = src + i * w_in;
            for (int j = 0 ; j < w_in; j++){
                unsigned char* outptr0 = dst + j * w_out + i;
                *outptr0 = *inptr0++;
            }
        }
    }
}
#endif
/*
1 2 3
4 5 6
7 8 9
rotate:
3 6 9
2 5 8
1 4 7
*/
//dst = (h_out - 1) * w_out
//类似rotate90，将输出结果倒着输出 或者先rotate90,然后沿Y轴翻转
#ifdef __aarch64__
void rotate270(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out){
    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int stride_h = 4 * w_in;
    int stride_h_w = 4 * w_in - 8;
    int hout = h_out - 1;
    //block 8*8. -- 8*8
    int i = 0;
    for (; i < h_in - 7; i += 8){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;
        // const unsigned char* inptr4 = inptr3 + w_in;
        // const unsigned char* inptr5 = inptr4 + w_in;
        // const unsigned char* inptr6 = inptr5 + w_in;
        // const unsigned char* inptr7 = inptr6 + w_in;

        asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
                "prfm   pldl1keep, [%[ptr0], #64]   \n"
                "prfm   pldl1keep, [%[ptr1]]        \n"
                "prfm   pldl1keep, [%[ptr1], #64]   \n"
                "prfm   pldl1keep, [%[ptr2]]        \n"
                "prfm   pldl1keep, [%[ptr2], #64]   \n"
                "prfm   pldl1keep, [%[ptr3]]        \n"
                "prfm   pldl1keep, [%[ptr3], #64]   \n"
                // "prfm   pldl1keep, [%[ptr4]]        \n"
                // "prfm   pldl1keep, [%[ptr4], #64]   \n"
                // "prfm   pldl1keep, [%[ptr5]]        \n"
                // "prfm   pldl1keep, [%[ptr5], #64]   \n"
        :
        :[ptr0] "r"(inptr0), [ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
        :"memory"
        );
        int j = 0;
        for (; j < w_in - 7; j += 8){
            unsigned char* outptr0 = dst + (hout - j) * w_out + i;
            unsigned char* outptr1 = outptr0 - w_out;
            unsigned char* outptr2 = outptr1 - w_out;
            unsigned char* outptr3 = outptr2 - w_out;
            unsigned char* outptr4 = outptr3 - w_out;
            unsigned char* outptr5 = outptr4 - w_out;
            unsigned char* outptr6 = outptr5 - w_out;
            unsigned char* outptr7 = outptr6 - w_out;
            // printf("outptr0: %x, inptr0: %x \n", outptr0, inptr0);
            // printf("inptr0: %d, %d, %d, %d \n", inptr0[0], inptr0[1], inptr0[2], inptr0[3]);
            asm volatile (
                "ld1  {v0.8b}, [%[inptr0]]    \n" //v0={00,01,02, 03, 04, 05, 06, 07}"
                "ld1  {v1.8b}, [%[inptr1]]    \n" //v0={10,11,12, 13, 14, 15, 16, 17}"
                "ld1  {v2.8b}, [%[inptr2]]    \n" //v0={20,21,22, 23, 24, 25, 26, 27}"
                "ld1  {v3.8b}, [%[inptr3]]    \n" //v0={30,31,32, 33, 34, 35, 36, 37}"

                "add %[inptr0], %[inptr0], %[stride_h] \n" //4 +4*w_in
                "add %[inptr1], %[inptr1], %[stride_h] \n" //5
                "add %[inptr2], %[inptr2], %[stride_h] \n" //6
                "add %[inptr3], %[inptr3], %[stride_h] \n" //7

                "trn1 v4.8b, v0.8b, v1.8b             \n" //v4={00 10 02 12 04 14 06 16 }
                "trn1 v6.8b, v2.8b, v3.8b             \n" //v4={20 30 22 32 24 34 26 36 }

                "trn2 v5.8b, v0.8b, v1.8b             \n" //v5={01 11 03 13 05 15 07 17 }
                "trn2 v7.8b, v2.8b, v3.8b             \n" //v7={21 31 23 33 25 35 27 37 }

                "ld1  {v12.8b}, [%[inptr0]]    \n" //v0={00,01,02, 03, 04, 05, 06, 07}"
                "ld1  {v13.8b}, [%[inptr1]]    \n" //v0={10,11,12, 13, 14, 15, 16, 17}"
                "ld1  {v14.8b}, [%[inptr2]]    \n" //v0={20,21,22, 23, 24, 25, 26, 27}"
                "ld1  {v15.8b}, [%[inptr3]]    \n" //v0={30,31,32, 33, 34, 35, 36, 37}"

                "trn1 v0.4h, v4.4h, v6.4h             \n" //v0={00 10 20 30 04 14 24 34}
                "trn1 v2.4h, v5.4h, v7.4h             \n" //v2={01 11 21 31 05 15 25 35}

                "trn2 v1.4h, v4.4h, v6.4h             \n" //v1={02 12 22 32 06 16 26 36}
                "trn2 v3.4h, v5.4h, v7.4h             \n" //v3={03 13 23 33 07 17 27 37}

                "trn1 v9.8b, v12.8b, v13.8b             \n" //v9={40 50 42 52 44 54 46 56 }
                "trn1 v11.8b, v14.8b, v15.8b             \n" //v11={60 70 62 72 64 74 66 76 }

                "trn2 v10.8b, v12.8b, v13.8b             \n" //v10={01 11 03 13 05 15 07 17 }
                "trn2 v12.8b, v14.8b, v15.8b             \n" //v12={21 31 23 33 25 35 27 37 }

                "sub %[inptr0], %[inptr0], %[stride_h_w] \n" //4 - 4*w_in + 8
                "sub %[inptr1], %[inptr1], %[stride_h_w] \n" //5
                "sub %[inptr2], %[inptr2], %[stride_h_w] \n" //6
                "sub %[inptr3], %[inptr3], %[stride_h_w] \n" //7

                "trn1 v4.4h, v9.4h, v11.4h             \n" //v4={40 50 60 70 44 54 64 74}
                "trn1 v6.4h, v10.4h, v12.4h             \n" //v6={41 51 61 71 45 55 65 75}

                "trn2 v5.4h, v9.4h, v11.4h             \n" //v5={42 52 62 72 46 56 66 76}
                "trn2 v7.4h, v10.4h, v12.4h            \n" //v7={43 53 63 73 47 57 67 77}

                "trn1 v8.2s, v0.2s, v4.2s             \n" //v8={00 10 20 30 40 50 60 70}
                "trn1 v9.2s, v2.2s, v6.2s             \n" //v6={01 11 21 31 41 51 61 71}
                "trn1 v10.2s, v1.2s, v5.2s             \n" //v10={02 12 22 32 42 52 62 72}
                "trn1 v11.2s, v3.2s, v7.2s             \n" //v11={03 13 23 33 43 53 63 73}

                "trn2 v12.2s, v0.2s, v4.2s             \n" //v12={04 14 24 34 44 54 64 74}
                "trn2 v13.2s, v2.2s, v6.2s             \n" //v13={05 15 25 35 45 55 65 75}
                "trn2 v14.2s, v1.2s, v5.2s             \n" //v14={06 16 26 36 46 56 66 76}
                "trn2 v15.2s, v3.2s, v7.2s             \n" //v15={07 17 27 37 47 57 67 77}

                "st1 {v8.8b}, [%[outptr0]], #8             \n" //00 10 20 30 04 14 24 34
                "st1 {v9.8b}, [%[outptr1]], #8              \n" //02 12 22 32
                "st1 {v10.8b}, [%[outptr2]], #8              \n" //01 11 21 31
                "st1 {v11.8b}, [%[outptr3]], #8              \n" //03 13 23 33

                "st1 {v12.8b}, [%[outptr4]], #8             \n" //00 10 20 30 04 14 24 34
                "st1 {v13.8b}, [%[outptr5]], #8              \n" //02 12 22 32
                "st1 {v14.8b}, [%[outptr6]], #8              \n" //01 11 21 31
                "st1 {v15.8b}, [%[outptr7]], #8              \n" //03 13 23 33

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), \
              [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5), [outptr6] "+r"(outptr6), [outptr7] "+r"(outptr7), \
              [stride_h] "+r"(stride_h), [stride_h_w] "+r"(stride_h_w)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",  "v12", "v13", "v14", "v15"
            );
            //  printf("outptr0: %x, inptr0: %x \n", outptr0, inptr0);
            // // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
            // outptr0 -= 8;
            // printf("outptr0: %d, %d, %d, %d \n", outptr0[0], outptr0[1], outptr0[2], outptr0[3]);
            // printf("outptr0: %d, %d, %d, %d \n", outptr0[4], outptr0[5], outptr0[6], outptr0[7]);
        }
        const unsigned char* inptr4 = inptr3 + w_in;
        const unsigned char* inptr5 = inptr4 + w_in;
        const unsigned char* inptr6 = inptr5 + w_in;
        const unsigned char* inptr7 = inptr6 + w_in;
        for (; j < w_in; j++){
            unsigned char* outptr = dst + (hout - j) * w_out + i;
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
            *outptr++ = *inptr6++;
            *outptr++ = *inptr7++;
        }

    }
    for (; i < h_in; i++){
        const unsigned char* inptr0 = src + i * w_in;
        for (int j = 0 ; j < w_in; j++){
            unsigned char* outptr0 = dst + (hout - j) * w_out + i;
            *outptr0 = *inptr0++;
        }
    }
}
#else
void rotate270(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out){

    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int hremain = h_in % 4;
    int hout = h_out - 1;
    //block 4*8. -- 8*4
    int i = 0;
    for (; i < h_in - 3; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;
        // const unsigned char* inptr4 = inptr3 + w_in;
        // const unsigned char* inptr5 = inptr4 + w_in;
        asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
                "pld [%[ptr0], #64]            @ preload a, 64byte\n"
                "pld [%[ptr1]]            @ preload a, 64byte\n"
                "pld [%[ptr1], #64]            @ preload a, 64byte\n"
                "pld [%[ptr2]]            @ preload a, 64byte\n"
                "pld [%[ptr2], #64]            @ preload a, 64byte\n"
                "pld [%[ptr3]]            @ preload a, 64byte\n"
                "pld [%[ptr3], #64]            @ preload a, 64byte\n"
                // "pld [%[ptr4]]            @ preload a, 64byte\n"
                // "pld [%[ptr4], #64]            @ preload a, 64byte\n"
                // "pld [%[ptr5]]            @ preload a, 64byte\n"
                // "pld [%[ptr5], #64]            @ preload a, 64byte\n"
        :
        :[ptr0] "r"(inptr0), [ptr1] "r"(inptr1), [ptr2] "r"(inptr2), [ptr3] "r"(inptr3)
       // [ptr4] "r"(inptr4),[ptr5] "r"(inptr5)
        :"memory"
        );
        int j = 0;
        for (; j < w_in - 7; j += 8){
           // printf("j: %d, inptr0: %x \n", j, inptr0);
            unsigned char* outptr0 = dst + (hout - j) * w_out + i;
            unsigned char* outptr1 = outptr0 - w_out;
            unsigned char* outptr2 = outptr1 - w_out;
            unsigned char* outptr3 = outptr2 - w_out;
            unsigned char* outptr4 = outptr3 - w_out;
            unsigned char* outptr5 = outptr4 - w_out;
            unsigned char* outptr6 = outptr5 - w_out;
            unsigned char* outptr7 = outptr6 - w_out;
            asm volatile (
                "vld1.8  {d0}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 04 05 06 07\n"
                "vld1.8  {d4}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 14 15 16 17\n"
                "vld1.8  {d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 24 25 26 27\n"
                "vld1.8  {d12}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 33 34 35 36 37\n"

                "vtrn.8  d0, d4                  @ trans data: d0=r00,r10,r02,r12 04 14 06 16  d4=01 11 03 13 \n"
                "vtrn.8  d8, d12                  @ trans data: d8=r20,r30,r12,r32 24 34 26 36  d12=21 31 23 33 \n"

                "vtrn.16 d0, d8                  @ trans data: d0=r00,r10,r20,r30 04 14 24 34 \n"
                "vtrn.16  d4, d12                  @ trans data: d4=01 11 21 31 05 15 25 35 \n"
                "vtrn.32 d0, d4                  @ trans data: d0=00,10,20,30 01 11 21 31 \n"
                "vtrn.32  d8, d12                  @ trans data: d8= 02 12 22 32 03 13 23 33\n"
                // "vtrn.16  d1, d9                   @ trans data: d1=r04,r14,r24,r34  d9=06 16 26 36 \n"

                "vst1.32  {d0[0]},    [%[outptr0]]!   @ write d0(q0,low),r00,r10 20 30\n"
                "vst1.32  {d0[1]},    [%[outptr1]]!   @ write d4(q0,low),r01,r11 21 31\n"
                "vst1.32  {d4[0]},    [%[outptr4]]!   @ write d2(q0,low),r02,r12\n"
                "vst1.32  {d4[1]},    [%[outptr5]]!   @ write d6(q0,low),r06,r16\n"
                "vst1.32  {d8[0]},    [%[outptr2]]!   @ write d1(q0,high),r02,r12 22 32\n"
                "vst1.32  {d8[1]},    [%[outptr3]]!   @ write d5(q0,low),r03,r13 23 33\n"
                "vst1.32 {d12[0]},     [%[outptr6]]!   @ write d3(q0,low),r03,r13\n"
                "vst1.32  {d12[1]},    [%[outptr7]]!   @ write d7(q0,low),r07,r17\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), \
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3), \
              [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5), [outptr6] "+r"(outptr6), [outptr7] "+r"(outptr7)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
           //  printf("j: %d, outptr0: %x \n", j, outptr0);
        }
        //printf("j: %d, inptr0: %x\n", j, inptr0);
        for (; j < w_in; j++){
            unsigned char* outptr = dst + (hout - j) * w_out + i;
           // printf("j: %d, outptr: %x \n", j, outptr);
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
        }

    }
    if (hremain > 0){
        for (; i < h_in; i++){
            const unsigned char* inptr0 = src + i * w_in;
            for (int j = 0 ; j < w_in; j++){
                unsigned char* outptr0 = dst + (hout - j) * w_out + i;
                *outptr0 = *inptr0++;
            }
        }
    }
}
#endif
/*
1 2 3
4 5 6
7 8 9
rotate:
3 2 1
6 5 4
9 8 7
*/
#ifdef __aarch64__
void rotate180(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out){
    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int stride_w = 8;
    //printf("dst: %x \n", dst);
    for (int i = 0; i < h_in; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;

        unsigned char* outptr0 = dst + (i + 1) * w_out - stride_w;//last
        unsigned char* outptr1 = outptr0 + w_out;
        unsigned char* outptr2 = outptr1 + w_out;
        unsigned char* outptr3 = outptr2 + w_out;

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
void rotate180(const unsigned char* src, unsigned char* dst, int w_in, int h_in, int w_out, int h_out){
    uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int stride_w = 8;
    //4*8
    //printf("dst: %x \n", dst);
    for (int i = 0; i < h_in; i += 4){
        const unsigned char* inptr0 = src + i * w_in;
        const unsigned char* inptr1 = inptr0 + w_in;
        const unsigned char* inptr2 = inptr1 + w_in;
        const unsigned char* inptr3 = inptr2 + w_in;

        unsigned char* outptr0 = dst + (i + 1) * w_out - stride_w;//last
        unsigned char* outptr1 = outptr0 + w_out;
        unsigned char* outptr2 = outptr1 + w_out;
        unsigned char* outptr3 = outptr2 + w_out;
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
