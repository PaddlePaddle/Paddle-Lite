#if 0
#define GEMM_SDOT_INT8_INTRAGROUP_KERNEL                                    \
  "ldp    q0, q1, [%[a_ptr]], #32\n"     /* load a00,a01,a10,a11 to q0, q1*/\
  "ldp    q2, q3, [%[i_ptr]], #32\n"     /* load b0, b1 to q4, q5*/         \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load i00,i01,i10,i11 to q4, q5*/\
  "eor    v8.16b,  v8.16b, v8.16b\n"     /* out0 = 0 */                    \
  "eor    v9.16b,  v9.16b, v9.16b\n"     /* out1 = 0 */                    \
  "eor    v10.16b,  v10.16b, v10.16b\n"  /* out2 = 0 */                    \
  "prfm   pldl1keep, [%[i_ptr], #64]\n"  /* preload a*/                    \
  "eor    v11.16b,  v11.16b, v11.16b\n"  /* out3 = 0 */                    \
  "eor    v12.16b,  v12.16b, v12.16b\n"  /* out4 = 0 */                    \
  "prfm   pldl1keep, [%[i_ptr], #128]\n" /* preload a*/                    \
  "eor    v13.16b,  v13.16b, v13.16b\n"  /* out5 = 0 */                    \
  "eor    v14.16b,  v14.16b, v14.16b\n"  /* out6 = 0 */                    \
  "prfm   pldl1keep, [%[i_ptr], #192]\n" /* preload a*/                    \
  "eor    v15.16b,  v15.16b, v15.16b\n"  /* out7 = 0 */                    \
  "eor    v16.16b,  v16.16b, v16.16b\n"  /* out8 = 0 */                    \
  "prfm   pldl1keep, [%[i_ptr], #256]\n" /* preload a*/                    \
  "eor    v17.16b,  v17.16b, v17.16b\n"  /* out9 = 0 */                    \
  "eor    v18.16b,  v18.16b, v18.16b\n"  /* out10 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"  /* preload b*/                    \
  "eor    v19.16b,  v19.16b, v19.16b\n"  /* out11 = 0 */                   \
  "eor    v20.16b,  v20.16b, v20.16b\n"  /* out12 = 0 */                   \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/                    \
  "eor    v21.16b,  v21.16b, v21.16b\n"  /* out13 = 0 */                   \
  "eor    v22.16b,  v22.16b, v22.16b\n"  /* out14 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #128]\n" /* preload b*/                    \
  "eor    v23.16b,  v23.16b, v23.16b\n"  /* out15 = 0 */                   \
  "eor    v24.16b,  v24.16b, v24.16b\n"  /* out16 = 0 */                   \
  "prfm   pldl1keep, [%[a_ptr], #128]\n" /* preload a*/                    \
  "eor    v25.16b,  v25.16b, v25.16b\n"  /* out17 = 0 */                   \
  "eor    v26.16b,  v26.16b, v26.16b\n"  /* out18 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #192]\n" /* preload b*/                    \
  "eor    v27.16b,  v27.16b, v27.16b\n"  /* out19 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #256]\n" /* preload b*/                    \
  "eor    v28.16b,  v28.16b, v28.16b\n"  /* out20 = 0 */                   \
  "prfm   pldl1keep, [%[a_ptr], #192]\n" /* preload a*/                    \
  "eor    v29.16b,  v29.16b, v29.16b\n"  /* out21 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #320]\n" /* preload b*/                    \
  "eor    v30.16b,  v30.16b, v30.16b\n"  /* out22 = 0 */                   \
  "prfm   pldl1keep, [%[a_ptr], #256]\n" /* preload a*/                    \
  "eor    v31.16b,  v31.16b, v31.16b\n"  /* out23 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #384]\n" /* preload b*/                    \
  "cbz    %w[k], 2f\n" /* check loop count > 0 */                          \
  /* main loop, unrool 0*/                                                 \
  "1:\n"                                 /* main loop */                   \
  "tbl    v6.16b, v4.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v4.16b, v3.16b\n"                                        \
  "prfm   pldl1keep, [%[b_ptr], #448]\n" /* preload a*/                    \
  "sdot   v8.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v20.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q4, [%[b_ptr]], #16   \n"                                        \
  "tbl    v6.16b, v5.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v5.16b, v3.16b\n"                                        \
  "sdot   v11.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v23.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q5, [%[b_ptr]], #16   \n"                                        \

  "tbl    v6.16b, v4.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v4.16b, v3.16b\n"                                        \
  "prfm   pldl1keep, [%[i_ptr], #320]\n" /* preload index*/                \
  "sdot   v14.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v26.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q4, [%[b_ptr]], #16   \n"                                        \
  "tbl    v6.16b, v5.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v5.16b, v3.16b\n"                                        \
  "sdot   v17.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v29.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q5, [%[b_ptr]], #16   \n"                                        \

  "tbl    v6.16b, v4.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v4.16b, v3.16b\n"                                        \
  "prfm   pldl1keep, [%[b_ptr], #448]\n" /* preload */                    \
  "sdot   v9.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v21.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q4, [%[b_ptr]], #16   \n"                                        \
  "tbl    v6.16b, v5.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v5.16b, v3.16b\n"                                        \
  "sdot   v12.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v24.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q5, [%[b_ptr]], #16   \n"                                        \

  "tbl    v6.16b, v4.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v4.16b, v3.16b\n"                                        \
  "prfm   pldl1keep, [%[a_ptr], #320]\n" /* preload a*/                    \
  "sdot   v15.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v27.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q4, [%[b_ptr]], #16   \n"                                        \
  "tbl    v6.16b, v5.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v5.16b, v3.16b\n"                                        \
  "sdot   v18.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v30.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q5, [%[b_ptr]], #16   \n"                                        \

  "tbl    v6.16b, v4.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v4.16b, v3.16b\n"                                        \
  "prfm   pldl1keep, [%[b_ptr], #448]\n" /* preload b*/                    \
  "sdot   v10.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v22.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q4, [%[b_ptr]], #16   \n"                                        \
  "tbl    v6.16b, v5.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v5.16b, v3.16b\n"                                        \
  "sdot   v13.4s,  v0.16b, v6.16b\n"                                        \
  "sdot   v25.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q5, [%[b_ptr]], #16   \n"                                        \

  "tbl    v6.16b, v4.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v4.16b, v3.16b\n"                                        \
  "sdot   v16.4s,  v0.16b, v6.16b\n"                                       \
  "sdot   v28.4s, v1.16b, v7.16b\n"                                        \
  "ldr    q4, [%[b_ptr]], #16   \n"                                        \
  "tbl    v6.16b, v5.16b, v2.16b\n"                                        \
  "tbl    v7.16b, v5.16b, v3.16b\n"                                        \
  "sdot   v19.4s,  v0.16b, v6.16b\n"                                       \
  "ldr    q5, [%[b_ptr]], #16   \n"                                        \
  "ldp    q2, q3, [%[i_ptr]], #32\n"                                       \
  "sdot   v31.4s, v1.16b, v7.16b\n"                                        \
  "subs   %w[k], %w[k], #1\n"           /* loop count - 1*/                \
  "ldp    q0, q1, [%[a_ptr]], #32\n"                                       \
  "bne    1b\n" /* Target to use when K is 1 or 2 */                       \
  "2:\n"                                             /* process scale*/     \



#endif