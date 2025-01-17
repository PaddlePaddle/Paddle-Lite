#pragma once

#ifdef __loongarch_sx
#include <lsxintrin.h>
#endif // __loongarch_sx
#ifdef __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx

#define XXL_INLINE extern __inline __attribute__((__gnu_inline__, __always_inline__, __artificial__))

typedef long long int __m64;

#define LSX_TRANSPOSE4_S(row0, row1, row2, row3)       \
do {                                                   \
  __m128i __r0 = (__m128i)(row0);                      \
  __m128i __r1 = (__m128i)(row1);                      \
  __m128i __r2 = (__m128i)(row2);                      \
  __m128i __r3 = (__m128i)(row3);                      \
  __m128i __t0 = __lsx_vilvl_w(__r1, __r0);            \
  __m128i __t1 = __lsx_vilvh_w(__r1, __r0);            \
  __m128i __t2 = __lsx_vilvl_w(__r3, __r2);            \
  __m128i __t3 = __lsx_vilvh_w(__r3, __r2);            \
  (row0) = (__m128)__lsx_vilvl_d(__t2, __t0);          \
  (row1) = (__m128)__lsx_vilvh_d(__t2, __t0);          \
  (row2) = (__m128)__lsx_vilvl_d(__t3, __t1);          \
  (row3) = (__m128)__lsx_vilvh_d(__t3, __t1);          \
} while (0)

#define MASKCOPY(dst, src, mask, cnt_limit, elem_type)              \
do {                                                                \
  char *__dst = (char *)dst;                                        \
  char *__src = (char *)src;                                        \
  unsigned int __sz = sizeof(elem_type);                            \
  for (int __i = 0; __i < cnt_limit; __i++) {                       \
    if (!(mask & (1 << __i)))                                       \
      continue;                                                     \
    __builtin_memcpy(__dst + __i * __sz, __src + __i * __sz, __sz); \
  }                                                                 \
} while (0)


#define LSX_SHUFFLE(a, b, c, d) ((a&3)*64+(b&3)*16+(c&3)*4+d)

#ifdef __loongarch_asx
XXL_INLINE __m256i __lasx_cvt_128_256(__m128i a) {
  __m256i _0;
  __builtin_memcpy(&_0, &a, sizeof(__m128i));
  return _0;
}
XXL_INLINE __m128i __lasx_cvt_256_128(__m256i a) {
  __m128i _0;
  __builtin_memcpy(&_0, &a, sizeof(__m128i));
  return _0;
}
XXL_INLINE __m256i __lasx_xvperm_q(__m256i a, __m256i b, int c) {
  switch (((c>>2)&12)|(c&3)) {
    case 0: return __lasx_xvpermi_q(a, b, 0);
    case 1: return __lasx_xvpermi_q(a, b, 1);
    case 2: return __lasx_xvpermi_q(a, b, 2);
    case 3: return __lasx_xvpermi_q(a, b, 3);
    case 4: return __lasx_xvpermi_q(a, b, 16);
    case 5: return __lasx_xvpermi_q(a, b, 17);
    case 6: return __lasx_xvpermi_q(a, b, 18);
    case 7: return __lasx_xvpermi_q(a, b, 19);
    case 8: return __lasx_xvpermi_q(a, b, 32);
    case 9: return __lasx_xvpermi_q(a, b, 33);
    case 10: return __lasx_xvpermi_q(a, b, 34);
    case 11: return __lasx_xvpermi_q(a, b, 35);
    case 12: return __lasx_xvpermi_q(a, b, 48);
    case 13: return __lasx_xvpermi_q(a, b, 49);
    case 14: return __lasx_xvpermi_q(a, b, 50);
    case 15: return __lasx_xvpermi_q(a, b, 51);
  }
  __builtin_unreachable();
}
#endif // __loongarch_asx

#define DEF_FCMP(pre, sub, typi, typo) \
XXL_INLINE typo __##pre##fcmp_xxx_##sub(typi a, typi b, const int c) { \
  switch (c&31) { \
    case 0: return __##pre##fcmp_ceq_##sub(a, b); \
    case 1: return __##pre##fcmp_slt_##sub(a, b); \
    case 2: return __##pre##fcmp_sle_##sub(a, b); \
    case 3: return __##pre##fcmp_cun_##sub(a, b); \
    case 4: return __##pre##fcmp_cune_##sub(a, b); \
    case 5: return __##pre##fcmp_sule_##sub(b, a); \
    case 6: return __##pre##fcmp_sult_##sub(b, a); \
    case 7: return __##pre##fcmp_cor_##sub(a, b); \
    case 8: return __##pre##fcmp_cueq_##sub(a, b); \
    case 9: return __##pre##fcmp_sult_##sub(a, b); \
    case 10: return __##pre##fcmp_sule_##sub(a, b); \
    case 11: return __##pre##fcmp_caf_##sub(a, b); \
    case 12: return __##pre##fcmp_cne_##sub(a, b); \
    case 13: return __##pre##fcmp_sle_##sub(b, a); \
    case 14: return __##pre##fcmp_slt_##sub(b, a); \
    case 15: return __##pre##xori_b(__##pre##fcmp_caf_##sub(a, b), 0xff); \
    case 16: return __##pre##fcmp_seq_##sub(a, b); \
    case 17: return __##pre##fcmp_clt_##sub(a, b); \
    case 18: return __##pre##fcmp_cle_##sub(a, b); \
    case 19: return __##pre##fcmp_sun_##sub(a, b); \
    case 20: return __##pre##fcmp_sune_##sub(a, b); \
    case 21: return __##pre##fcmp_cule_##sub(b, a); \
    case 22: return __##pre##fcmp_cult_##sub(b, a); \
    case 23: return __##pre##fcmp_sor_##sub(a, b); \
    case 24: return __##pre##fcmp_sueq_##sub(a, b); \
    case 25: return __##pre##fcmp_cult_##sub(a, b); \
    case 26: return __##pre##fcmp_cule_##sub(a, b); \
    case 27: return __##pre##fcmp_saf_##sub(a, b); \
    case 28: return __##pre##fcmp_sne_##sub(a, b); \
    case 29: return __##pre##fcmp_cle_##sub(b, a); \
    case 30: return __##pre##fcmp_clt_##sub(b, a); \
    case 31: return __##pre##xori_b(__##pre##fcmp_saf_##sub(a, b), 0xff); \
  } \
  __builtin_unreachable(); \
}
#ifdef __loongarch_asx
DEF_FCMP(lasx_xv, d, __m256d, __m256i)
DEF_FCMP(lasx_xv, s, __m256, __m256i)
#endif // __loongarch_asx
#ifdef __loongarch_sx
DEF_FCMP(lsx_v, d, __m128d, __m128i)
DEF_FCMP(lsx_v, s, __m128, __m128i)
#endif // __loongarch_sx

#if defined(__loongarch_asx) && defined(__loongarch_sx)
XXL_INLINE __m128 lasx_extractf128_f32(__m256 a, const int b) {
  return (__m128)__lasx_cvt_256_128(b&1?__lasx_xvpermi_q((__m256i)a, (__m256i)a, 1):(__m256i)a);
}

XXL_INLINE __m128i lasx_extractf128_m256i(__m256i a, const int b) {
  return __lasx_cvt_256_128(b&1?__lasx_xvpermi_q(a, a, 1):a);
}

XXL_INLINE __m256 lasx_insertf128_f32(__m256 a, __m128 b, int c) {
  __m256 _0;
  _0 = (__m256)__lasx_xvpermi_q(a, __lasx_cvt_128_256((__m128i)b), c & 1 ? 2 : 48);
  return _0;
}

XXL_INLINE __m128 lasx_castm256_m128(__m256 a) {
  return (__m128)__lasx_cvt_256_128((__m256i)a);
}

XXL_INLINE __m128d lasx_castm256d_m128d(__m256d a) {
  return (__m128d)__lasx_cvt_256_128((__m256i)a);
}

XXL_INLINE __m128i lasx_castm256i_m128i(__m256i a) {
  return __lasx_cvt_256_128(a);
}

XXL_INLINE __m256i lasx_broadcastm128i_m256i(__m128i a) {
  return __lasx_xvreplve0_q(__lasx_cvt_128_256(a));
}

XXL_INLINE __m128i lasx_extracti128_m256i(__m256i a, const int b) {
  return __lasx_cvt_256_128(b&1?__lasx_xvpermi_q(a, a, 1):a);
}

XXL_INLINE __m256i lasx_inserti128_m256i(__m256i a, __m128i b, const int c) {
  __m256i _0;
  _0 = __lasx_xvpermi_q(a, __lasx_cvt_128_256(b), c & 1 ? 2 : 48);
  return _0;
}

XXL_INLINE __m256i lasx_cvti8_i16(__m128i a) {
  return __lasx_vext2xv_h_b(__lasx_cvt_128_256(a));
}

XXL_INLINE __m256i lasx_cvti8_i32(__m128i a) {
  return __lasx_vext2xv_w_b(__lasx_cvt_128_256(a));
}

#endif // __loongarch_asx and __loongarch_sx

#ifdef __loongarch_asx
XXL_INLINE __m256d lasx_add_f64(__m256d a, __m256d b) {
  return __lasx_xvfadd_d(a, b);
}

XXL_INLINE __m256 lasx_add_f32(__m256 a, __m256 b) {
  return __lasx_xvfadd_s(a, b);
}

XXL_INLINE __m256d lasx_div_f64(__m256d a, __m256d b) {
  return __lasx_xvfdiv_d(a, b);
}

XXL_INLINE __m256 lasx_div_f32(__m256 a, __m256 b) {
  return __lasx_xvfdiv_s(a, b);
}

XXL_INLINE __m256d lasx_hadd_f64(__m256d a, __m256d b) {
  return __lasx_xvfadd_d((__m256d)__lasx_xvpickev_d((__m256i)b, (__m256i)a), (__m256d)__lasx_xvpickod_d((__m256i)b, (__m256i)a));
}

XXL_INLINE __m256 lasx_hadd_f32(__m256 a, __m256 b) {
  return __lasx_xvfadd_s((__m256)__lasx_xvpickev_w((__m256i)b, (__m256i)a), (__m256)__lasx_xvpickod_w((__m256i)b, (__m256i)a));
}

XXL_INLINE __m256d lasx_mul_f64(__m256d a, __m256d b) {
  return __lasx_xvfmul_d(a, b);
}

XXL_INLINE __m256 lasx_mul_f32(__m256 a, __m256 b) {
  return __lasx_xvfmul_s(a, b);
}

XXL_INLINE __m256d lasx_sub_f64(__m256d a, __m256d b) {
  return __lasx_xvfsub_d(a, b);
}

XXL_INLINE __m256 lasx_sub_f32(__m256 a, __m256 b) {
  return __lasx_xvfsub_s(a, b);
}

XXL_INLINE __m256 lasx_and_f32(__m256 a, __m256 b) {
  return (__m256)__lasx_xvand_v((__m256i)a, (__m256i)b);
}

XXL_INLINE __m256 lasx_andnot_f32(__m256 a, __m256 b) {
  return (__m256)__lasx_xvandn_v((__m256i)a, (__m256i)b);
}

XXL_INLINE __m256 lasx_or_f32(__m256 a, __m256 b) {
  return (__m256)__lasx_xvor_v((__m256i)a, (__m256i)b);
}

XXL_INLINE __m256 lasx_xor_f32(__m256 a, __m256 b) {
  return (__m256)__lasx_xvxor_v((__m256i)a, (__m256i)b);
}

XXL_INLINE __m256 lasx_blend_f32(__m256 a, __m256 b, const int c) {
  return (__m256)__lasx_xvbitsel_v((__m256i)a, (__m256i)b, __lasx_vext2xv_w_b(__lasx_xvldi(c|0xf900)));
}

XXL_INLINE __m256 lasx_blendv_f32(__m256 a, __m256 b, __m256 c) {
  return (__m256)__lasx_xvbitsel_v((__m256i)a, (__m256i)b, __lasx_xvslti_w((__m256i)c, 0));
}

XXL_INLINE __m256 lasx_shuffle_f32(__m256 a, __m256 b, const int c) {
  return (__m256)__lasx_xvpermi_w((__m256i)b, (__m256i)a, (unsigned int)c);
}

XXL_INLINE __m256 lasx_permute2f128_f32(__m256 a, __m256 b, int c) {
  __m256 _0;
  __m256i _1 = __lasx_xvldi(0);
  __m256i _2 = c&136 ? _1 : (__m256i)b;
  __m256 _3 = c&128 ? (c&2?b:a) : c&0x8 ? (c&32?b:a) : a;
  int _4 = c&128 ? ((c&1)|32) : c&0x8 ? ((c&16)|2) : c&51;
  _0 = (c&136)==136 ? (__m256)_1 : (__m256)__lasx_xvperm_q(_2, (__m256i)_3, _4);
  return _0;
}

XXL_INLINE __m256d lasx_permute2f128_f64(__m256d a, __m256d b, int c) {
  __m256d _0;
  __m256i _1 = __lasx_xvldi(0);
  __m256i _2 = c&136 ? _1 : (__m256i)b;
  __m256d _3 = c&128 ? (c&2?b:a) : c&0x8 ? (c&32?b:a) : a;
  int _4 = c&128 ? ((c&1)|32) : c&0x8 ? ((c&16)|2) : c&51;
  _0 = (c&136)==136 ? (__m256d)_1 : (__m256d)__lasx_xvperm_q(_2, (__m256i)_3, _4);
  return _0;
}

XXL_INLINE __m256 lasx_unpackhi_f32(__m256 a, __m256 b) {
  return (__m256)__lasx_xvilvh_w((__m256i)b, (__m256i)a);
}

XXL_INLINE __m256 lasx_unpacklo_f32(__m256 a, __m256 b) {
  return (__m256)__lasx_xvilvl_w((__m256i)b, (__m256i)a);
}

XXL_INLINE __m256d lasx_max_f64(__m256d a, __m256d b) {
  return __lasx_xvfmax_d(a, b);
}

XXL_INLINE __m256 lasx_max_f32(__m256 a, __m256 b) {
  return __lasx_xvfmax_s(a, b);
}

XXL_INLINE __m256 lasx_min_f32(__m256 a, __m256 b) {
  return __lasx_xvfmin_s(a, b);
}

XXL_INLINE __m256 lasx_floor_f32(__m256 a) {
  return __lasx_xvfrintrm_s(a);
}

XXL_INLINE __m256 lasx_cmp_f32(__m256 a, __m256 b, const int c) {
  return (__m256)__lasx_xvfcmp_xxx_s(a, b, c);
}

XXL_INLINE __m256 lasx_xvfcmp_slt_s(__m256 a, __m256 b) {
  return (__m256)__lasx_xvfcmp_slt_s(a, b);
}

XXL_INLINE __m256 lasx_xvfcmp_sle_s(__m256 a, __m256 b) {
  return (__m256)__lasx_xvfcmp_sle_s(a, b);
}

XXL_INLINE __m256 lasx_cvti32_f32(__m256i a) {
  return __lasx_xvffint_s_w(a);
}

XXL_INLINE __m256i lasx_cvtf32_i32(__m256 a) {
  return __lasx_xvftint_w_s(a);
}

XXL_INLINE __m256i lasx_cvttf32_i32(__m256 a) {
  return __lasx_xvftintrz_w_s(a);
}

XXL_INLINE __m256 lasx_broadcast_1f32(float const * a) {
  return (__m256)__lasx_xvldrepl_w((void *)a, 0);
}

XXL_INLINE __m256d lasx_broadcast_sd(double const * a) {
  return (__m256d)__lasx_xvldrepl_d((void *)a, 0);
}

XXL_INLINE __m256 lasx_load_f32(float const * a) {
  return (__m256)__lasx_xvld((void *)a, 0);
}

XXL_INLINE __m256d lasx_loadu_f64(double const * a) {
  return (__m256d)__lasx_xvld((void *)a, 0);
}

XXL_INLINE __m256 lasx_loadu_f32(float const * a) {
  return (__m256)__lasx_xvld((void *)a, 0);
}

XXL_INLINE __m256i lasx_loadu_m256i(__m256i const * a) {
  return __lasx_xvld((void *)a, 0);
}

XXL_INLINE void lasx_storeu_f64(double * a, __m256d b) {
  return __lasx_xvst((__m256i)b, (void *)a, 0);
}

XXL_INLINE void lasx_storeu_f32(float * a, __m256 b) {
  return __lasx_xvst((__m256i)b, (void *)a, 0);
}

XXL_INLINE void lasx_storeu_m256i(__m256i * a, __m256i b) {
  return __lasx_xvst(b, (void *)a, 0);
}

XXL_INLINE void lasx_maskstore_f32(float * a, __m256i b, __m256 c) {
  __m256i vmask = __lasx_xvmskltz_w(b);
  int mask = __lasx_xvpickve2gr_w(vmask, 0) | (__lasx_xvpickve2gr_w(vmask, 4) << 4);
  MASKCOPY(a, &c, mask, 8, float);
}

XXL_INLINE __m256d lasx_sqrt_f64(__m256d a) {
  return __lasx_xvfsqrt_d(a);
}

XXL_INLINE __m256 lasx_sqrt_f32(__m256 a) {
  return __lasx_xvfsqrt_s(a);
}

XXL_INLINE __m256d lasx_setzero_f64() {
  return (__m256d)__lasx_xvldi(0);
}

XXL_INLINE __m256 lasx_setzero_f32() {
  return (__m256)__lasx_xvldi(0);
}

XXL_INLINE __m256i lasx_setzero_m256i() {
  return __lasx_xvldi(0);
}

XXL_INLINE __m256 lasx_set_f32(float a, float b, float c, float d, float e, float f, float g, float h) {
  return (__m256)(v8f32){h,g,f,e,d,c,b,a};
}

XXL_INLINE __m256i lasx_set_i32(int a, int b, int c, int d, int e, int f, int g, int h) {
  return (__m256i)(v8i32){h,g,f,e,d,c,b,a};
}

XXL_INLINE __m256d lasx_set1_f64(double a) {
  return (__m256d)(v4f64){a,a,a,a};
}

XXL_INLINE __m256 lasx_set1_f32(float a) {
  return (__m256)(v8f32){a,a,a,a,a,a,a,a};
}

XXL_INLINE __m256i lasx_set1_i8(char a) {
  return __lasx_xvreplgr2vr_b((int)a);
}

XXL_INLINE __m256i lasx_set1_i16(short a) {
  return __lasx_xvreplgr2vr_h((int)a);
}

XXL_INLINE __m256i lasx_set1_i32(int a) {
  return __lasx_xvreplgr2vr_w(a);
}

XXL_INLINE __m256i lasx_set1_i64x(long long a) {
  return __lasx_xvreplgr2vr_d((long int)a);
}

XXL_INLINE __m256 lasx_castf64_f32(__m256d a) {
  return (__m256)a;
}

XXL_INLINE __m256d lasx_castf32_f64(__m256 a) {
  return (__m256d)a;
}

XXL_INLINE __m256i lasx_castf32_m256i(__m256 a) {
  return (__m256i)a;
}

XXL_INLINE __m256 lasx_castm256i_f32(__m256i a) {
  return (__m256)a;
}

XXL_INLINE __m256i lasx_permute2x128_m256i(__m256i a, __m256i b, const int c) {
  __m256i _0;
  __m256i _1 = __lasx_xvldi(0);
  __m256i _2 = c&136 ? _1 : b;
  __m256i _3 = c&128 ? (c&2?b:a) : c&0x8 ? (c&32?b:a) : a;
  int _4 = c&128 ? ((c&1)|32) : c&0x8 ? ((c&16)|2) : c&51;
  _0 = (c&136)==136 ? _1 : __lasx_xvperm_q(_2, _3, _4);
  return _0;
}

XXL_INLINE __m256i lasx_permute4x64_i64(__m256i a, const int b) {
  return __lasx_xvpermi_d(a, (unsigned int)b);
}

XXL_INLINE __m256d lasx_permute4x64_f64(__m256d a, const int b) {
  return (__m256d)__lasx_xvpermi_d((__m256i)a, (unsigned int)b);
}

XXL_INLINE __m256 lasx_permutevar8x32_f32(__m256 a, __m256i b) {
  return (__m256)__lasx_xvshuf_w(b, __lasx_xvpermi_q((__m256i)a, (__m256i)a, 17), __lasx_xvpermi_q((__m256i)a, (__m256i)a, 0));
}

XXL_INLINE __m256i lasx_unpackhi_i8(__m256i a, __m256i b) {
  return __lasx_xvilvh_b(b, a);
}

XXL_INLINE __m256i lasx_unpackhi_i16(__m256i a, __m256i b) {
  return __lasx_xvilvh_h(b, a);
}

XXL_INLINE __m256i lasx_unpackhi_i32(__m256i a, __m256i b) {
  return __lasx_xvilvh_w(b, a);
}

XXL_INLINE __m256i lasx_unpacklo_i8(__m256i a, __m256i b) {
  return __lasx_xvilvl_b(b, a);
}

XXL_INLINE __m256i lasx_unpacklo_i16(__m256i a, __m256i b) {
  return __lasx_xvilvl_h(b, a);
}

XXL_INLINE __m256i lasx_unpacklo_i32(__m256i a, __m256i b) {
  return __lasx_xvilvl_w(b, a);
}

XXL_INLINE __m256i lasx_max_i8(__m256i a, __m256i b) {
  return __lasx_xvmax_b(a, b);
}

XXL_INLINE __m256i lasx_max_i32(__m256i a, __m256i b) {
  return __lasx_xvmax_w(a, b);
}

XXL_INLINE __m256i lasx_min_i32(__m256i a, __m256i b) {
  return __lasx_xvmin_w(a, b);
}

XXL_INLINE __m256i lasx_add_i32(__m256i a, __m256i b) {
  return __lasx_xvadd_w(a, b);
}

XXL_INLINE __m256i lasx_add_i64(__m256i a, __m256i b) {
  return __lasx_xvadd_d(a, b);
}

XXL_INLINE __m256i lasx_adds_i16(__m256i a, __m256i b) {
  return __lasx_xvsadd_h(a, b);
}

XXL_INLINE __m256i lasx_hadd_i32(__m256i a, __m256i b) {
  return __lasx_xvadd_w(__lasx_xvpickev_w(b, a), __lasx_xvpickod_w(b, a));
}

XXL_INLINE __m256i lasx_madd_i16(__m256i a, __m256i b) {
  return __lasx_xvadd_w(__lasx_xvmulwev_w_h(a, b), __lasx_xvmulwod_w_h(a, b));
}

XXL_INLINE __m256i lasx_maddubs_i16(__m256i a, __m256i b) {
  return __lasx_xvsadd_h(__lasx_xvmulwev_h_bu_b(a, b), __lasx_xvmulwod_h_bu_b(a, b));
}

XXL_INLINE __m256i lasx_mullo_i32(__m256i a, __m256i b) {
  return __lasx_xvmul_w(a, b);
}

XXL_INLINE __m256i lasx_sub_i32(__m256i a, __m256i b) {
  return __lasx_xvsub_w(a, b);
}

XXL_INLINE __m256i lasx_sub_i64(__m256i a, __m256i b) {
  return __lasx_xvsub_d(a, b);
}

XXL_INLINE __m256i lasx_packs_i16(__m256i a, __m256i b) {
  return __lasx_xvpickev_b(__lasx_xvsat_h(b, 7), __lasx_xvsat_h(a, 7));
}

XXL_INLINE __m256i lasx_packs_i32(__m256i a, __m256i b) {
  return __lasx_xvpickev_h(__lasx_xvsat_w(b, 15), __lasx_xvsat_w(a, 15));
}

XXL_INLINE __m256i lasx_packus_i16(__m256i a, __m256i b) {
  return __lasx_xvpickev_b(__lasx_xvsat_hu(__lasx_xvmax_h(b, __lasx_xvldi(0)), 7), __lasx_xvsat_hu(__lasx_xvmax_h(a, __lasx_xvldi(0)), 7));
}

XXL_INLINE __m256i lasx_and_m256i(__m256i a, __m256i b) {
  return __lasx_xvand_v(a, b);
}

XXL_INLINE __m256i lasx_andnot_m256i(__m256i a, __m256i b) {
  return __lasx_xvandn_v(a, b);
}

XXL_INLINE __m256i lasx_or_m256i(__m256i a, __m256i b) {
  return __lasx_xvor_v(a, b);
}

XXL_INLINE __m256i lasx_cmpeq_i32(__m256i a, __m256i b) {
  return __lasx_xvseq_w(a, b);
}

XXL_INLINE __m256i lasx_maskload_i32(int const* a, __m256i b) {
  __m256i _0;
  _0 = __lasx_xvldi(0);
  __m256i vmask = __lasx_xvmskltz_w(b);
  int mask = __lasx_xvpickve2gr_w(vmask, 0) | (__lasx_xvpickve2gr_w(vmask, 4) << 4);
  MASKCOPY(&_0, a, mask, 8, int);
  return _0;
}

XXL_INLINE __m256i lasx_maskload_i64(long long const* a, __m256i b) {
  __m256i _0;
  _0 = __lasx_xvldi(0);
  __m256i vmask = __lasx_xvmskltz_d(b);
  int mask = __lasx_xvpickve2gr_w(vmask, 0) | (__lasx_xvpickve2gr_w(vmask, 4) << 2);
  MASKCOPY(&_0, a, mask, 4, long long int);
  return _0;
}

XXL_INLINE void lasx_maskstore_i32(int* a, __m256i b, __m256i c) {
  __m256i vmask = __lasx_xvmskltz_w(b);
  int mask = __lasx_xvpickve2gr_w(vmask, 0) | (__lasx_xvpickve2gr_w(vmask, 4) << 4);
  MASKCOPY(a, &c, mask, 8, int);
}

XXL_INLINE __m256i lasx_slli_i32(__m256i a, int b) {
  return b>=32?__lasx_xvldi(0):__lasx_xvslli_w(a, (unsigned int)b);
}

XXL_INLINE __m256i lasx_srli_i32(__m256i a, int b) {
  return b>=32?__lasx_xvldi(0):__lasx_xvsrli_w(a, (unsigned int)b);
}

XXL_INLINE __m256 lasx_fmadd_f32(__m256 a, __m256 b, __m256 c) {
  return __lasx_xvfmadd_s(a, b, c);
}

#endif // __loongarch_asx

#ifdef __loongarch_sx
XXL_INLINE __m128 lsx_cmp_f32(__m128 a, __m128 b, const int c) {
  return (__m128)__lsx_vfcmp_xxx_s(a, b, c);
}

XXL_INLINE __m128 lsx_vfcmp_slt_s(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_slt_s(a, b);
}

XXL_INLINE void lsx_maskstore_f32(float * a, __m128i b, __m128 c) {
  int mask = __lsx_vpickve2gr_w(__lsx_vmskltz_w(b), 0);
  MASKCOPY(a, &c, mask, 4, float);
}

XXL_INLINE __m128 lsx_fmadd_f32(__m128 a, __m128 b, __m128 c) {
  return __lsx_vfmadd_s(a, b, c);
}

XXL_INLINE __m128 lsx_shuffle_f32(__m128 a, __m128 b, unsigned int c) {
  return (__m128)__lsx_vpermi_w((__m128i)b, (__m128i)a, c);
}

XXL_INLINE __m128 lsx_unpackhi_f32(__m128 a, __m128 b) {
  return (__m128)__lsx_vilvh_w((__m128i)b, (__m128i)a);
}

XXL_INLINE __m128 lsx_unpacklo_f32(__m128 a, __m128 b) {
  return (__m128)__lsx_vilvl_w((__m128i)b, (__m128i)a);
}

XXL_INLINE __m128 lsx_min_f32(__m128 a, __m128 b) {
  return __lsx_vfmin_s(a, b);
}

XXL_INLINE __m128 lsx_max_f32(__m128 a, __m128 b) {
  return __lsx_vfmax_s(a, b);
}

XXL_INLINE __m128 lsx_add_f32(__m128 a, __m128 b) {
  return __lsx_vfadd_s(a, b);
}

XXL_INLINE __m128 lsx_sub_f32(__m128 a, __m128 b) {
  return __lsx_vfsub_s(a, b);
}

XXL_INLINE __m128 lsx_mul_f32(__m128 a, __m128 b) {
  return __lsx_vfmul_s(a, b);
}

XXL_INLINE __m128 lsx_div_f32(__m128 a, __m128 b) {
  return __lsx_vfdiv_s(a, b);
}

XXL_INLINE void lsx_storel_pi(__m64* a, __m128 b) {
  return __lsx_vstelm_d((__m128i)b, (void *)a, 0, 0);
}

XXL_INLINE void lsx_store_1f32(float* a, __m128 b) {
  return __lsx_vstelm_w((__m128i)b, (void *)a, 0, 0);
}

XXL_INLINE void lsx_storeu_f32(float* a, __m128 b) {
  return __lsx_vst((__m128i)b, (void *)a, 0);
}

XXL_INLINE __m128 lsx_sqrt_f32(__m128 a) {
  return __lsx_vfsqrt_s(a);
}

XXL_INLINE __m128 lsx_and_f32(__m128 a, __m128 b) {
  return (__m128)__lsx_vand_v((__m128i)a, (__m128i)b);
}

XXL_INLINE __m128 lsx_cmplt_f32(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_clt_s(a, b);
}

XXL_INLINE __m128 lsx_cmple_f32(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_cle_s(a, b);
}

XXL_INLINE __m128 lsx_cmpgt_f32(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_clt_s(b, a);
}

XXL_INLINE __m128 lsx_cmpge_f32(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_cle_s(b, a);
}

XXL_INLINE __m128 lsx_set1_f32(float a) {
  return (__m128)(v4f32){a,a,a,a};
}

XXL_INLINE __m128 lsx_setzero_f32() {
  return (__m128)__lsx_vldi(0);
}

XXL_INLINE __m128 lsx_loadl_pi(__m128 a, __m64 const* b) {
  __m128 _0;
  _0 = (__m128)a; __builtin_memcpy(&_0, b, sizeof(__m64));
  return _0;
}

XXL_INLINE __m128 lsx_load1_f32(float const* a) {
  return (__m128)__lsx_vldrepl_w((void *)a, 0);
}

XXL_INLINE __m128 lsx_loadu_f32(float const* a) {
  return (__m128)__lsx_vld((void *)a, 0);
}

XXL_INLINE __m128 lsx_movehl_f32(__m128 a, __m128 b) {
  return (__m128)__lsx_vilvh_d((__m128i)a, (__m128i)b);
}

XXL_INLINE __m128 lsx_movelh_f32(__m128 a, __m128 b) {
  return (__m128)__lsx_vilvl_d((__m128i)b, (__m128i)a);
}

XXL_INLINE __m128i lsx_loadl_i64(__m128i const* a) {
  __m128i _0;
  _0 = __lsx_vinsgr2vr_d(__lsx_vldi(0), *(long*)(a), 0);
  return _0;
}

XXL_INLINE __m128i lsx_loadu_m128i(__m128i const* a) {
  return __lsx_vld((void *)a, 0);
}

XXL_INLINE __m128d lsx_load1_f64(double const* a) {
  return (__m128d)__lsx_vldrepl_d((void *)a, 0);
}

XXL_INLINE __m128d lsx_loadu_f64(double const* a) {
  return (__m128d)__lsx_vld((void *)a, 0);
}

XXL_INLINE void lsx_storeu_m128i(__m128i* a, __m128i b) {
  return __lsx_vst(b, (void *)a, 0);
}

XXL_INLINE void lsx_storel_i64(__m128i* a, __m128i b) {
  return __lsx_vstelm_d(b, (void *)a, 0, 0);
}

XXL_INLINE void lsx_store_sd(double* a, __m128d b) {
  __builtin_memcpy(a, &b, sizeof(double));
}

XXL_INLINE void lsx_storeu_f64(double* a, __m128d b) {
  return __lsx_vst((__m128i)b, (void *)a, 0);
}

XXL_INLINE __m128i lsx_add_i32(__m128i a, __m128i b) {
  return __lsx_vadd_w(a, b);
}

XXL_INLINE __m128i lsx_add_i64(__m128i a, __m128i b) {
  return __lsx_vadd_d(a, b);
}

XXL_INLINE __m128i lsx_madd_i16(__m128i a, __m128i b) {
  return __lsx_vadd_w(__lsx_vmulwev_w_h(a, b), __lsx_vmulwod_w_h(a, b));
}

XXL_INLINE __m128i lsx_sub_i32(__m128i a, __m128i b) {
  return __lsx_vsub_w(a, b);
}

XXL_INLINE __m128i lsx_sub_i64(__m128i a, __m128i b) {
  return __lsx_vsub_d(a, b);
}

XXL_INLINE __m128d lsx_add_f64(__m128d a, __m128d b) {
  return __lsx_vfadd_d(a, b);
}

XXL_INLINE __m128d lsx_div_f64(__m128d a, __m128d b) {
  return __lsx_vfdiv_d(a, b);
}

XXL_INLINE __m128d lsx_mul_f64(__m128d a, __m128d b) {
  return __lsx_vfmul_d(a, b);
}

XXL_INLINE __m128d lsx_sub_f64(__m128d a, __m128d b) {
  return __lsx_vfsub_d(a, b);
}

XXL_INLINE __m128d lsx_max_f64(__m128d a, __m128d b) {
  return __lsx_vfmax_d(a, b);
}

XXL_INLINE __m128i lsx_srli_m128i(__m128i a, int b) {
  return b>=16?__lsx_vldi(0):__lsx_vbsrl_v(a, (unsigned int)b);
}

XXL_INLINE __m128 lsx_cvti32_f32(__m128i a) {
  return __lsx_vffint_s_w(a);
}

XXL_INLINE __m128i lsx_cvtf32_i32(__m128 a) {
  return __lsx_vftint_w_s(a);
}

XXL_INLINE __m128i lsx_set_i8(char a, char b, char c, char d, char e, char f, char g, char h, char i, char j, char k, char l, char m, char n, char o, char p) {
  return (__m128i)(v16i8){p,o,n,m,l,k,j,i,h,g,f,e,d,c,b,a};
}

XXL_INLINE __m128i lsx_set1_i64(__m64 a) {
  return __lsx_vreplgr2vr_d((long int)a);
}

XXL_INLINE __m128i lsx_set1_i64x(long long a) {
  return __lsx_vreplgr2vr_d((long int)a);
}

XXL_INLINE __m128i lsx_set1_i32(int a) {
  return __lsx_vreplgr2vr_w(a);
}

XXL_INLINE __m128i lsx_set1_i16(short a) {
  return __lsx_vreplgr2vr_h((int)a);
}

XXL_INLINE __m128i lsx_set1_i8(char a) {
  return __lsx_vreplgr2vr_b((int)a);
}

XXL_INLINE __m128i lsx_setr_i32(int a, int b, int c, int d) {
  return (__m128i)(v4i32){a,b,c,d};
}

XXL_INLINE __m128i lsx_setzero_m128i() {
  return __lsx_vldi(0);
}

XXL_INLINE __m128d lsx_set1_f64(double a) {
  return (__m128d)(v2f64){a,a};
}

XXL_INLINE __m128d lsx_setzero_f64() {
  return (__m128d)__lsx_vldi(0);
}

XXL_INLINE __m128i lsx_packs_i16(__m128i a, __m128i b) {
  return __lsx_vpickev_b(__lsx_vsat_h(b, 7), __lsx_vsat_h(a, 7));
}

XXL_INLINE __m128i lsx_packs_i32(__m128i a, __m128i b) {
  return __lsx_vpickev_h(__lsx_vsat_w(b, 15), __lsx_vsat_w(a, 15));
}

XXL_INLINE __m128i lsx_packus_i16(__m128i a, __m128i b) {
  return __lsx_vpickev_b(__lsx_vsat_hu(__lsx_vmax_h(b, __lsx_vldi(0)), 7), __lsx_vsat_hu(__lsx_vmax_h(a, __lsx_vldi(0)), 7));
}

XXL_INLINE __m128i lsx_shuffle_i32(__m128i a, int b) {
  return __lsx_vshuf4i_w(a, (unsigned int)b);
}

XXL_INLINE __m128i lsx_unpackhi_i8(__m128i a, __m128i b) {
  return __lsx_vilvh_b(b, a);
}

XXL_INLINE __m128i lsx_unpackhi_i16(__m128i a, __m128i b) {
  return __lsx_vilvh_h(b, a);
}

XXL_INLINE __m128i lsx_unpackhi_i32(__m128i a, __m128i b) {
  return __lsx_vilvh_w(b, a);
}

XXL_INLINE __m128i lsx_unpacklo_i8(__m128i a, __m128i b) {
  return __lsx_vilvl_b(b, a);
}

XXL_INLINE __m128i lsx_unpacklo_i16(__m128i a, __m128i b) {
  return __lsx_vilvl_h(b, a);
}

XXL_INLINE __m128i lsx_unpacklo_i32(__m128i a, __m128i b) {
  return __lsx_vilvl_w(b, a);
}

XXL_INLINE __m128d lsx_sqrt_f64(__m128d a) {
  return __lsx_vfsqrt_d(a);
}

XXL_INLINE __m128i lsx_castf32_m128i(__m128 a) {
  return (__m128i)a;
}

XXL_INLINE __m128 lsx_castm128i_f32(__m128i a) {
  return (__m128)a;
}

XXL_INLINE __m128d lsx_hadd_f64(__m128d a, __m128d b) {
  return __lsx_vfadd_d((__m128d)__lsx_vpickev_d((__m128i)b, (__m128i)a), (__m128d)__lsx_vpickod_d((__m128i)b, (__m128i)a));
}

XXL_INLINE __m128 lsx_hadd_f32(__m128 a, __m128 b) {
  return __lsx_vfadd_s((__m128)__lsx_vpickev_w((__m128i)b, (__m128i)a), (__m128)__lsx_vpickod_w((__m128i)b, (__m128i)a));
}

XXL_INLINE __m128 lsx_blend_f32(__m128 a, __m128 b, const int c) {
  return (__m128)__lsx_vbitsel_v((__m128i)a, (__m128i)b, __lsx_vilvl_h(__lsx_vldi((c&1)|(c&1)<<1|(c&2)<<1|(c&2)<<2|(c&4)<<2|(c&4)<<3|(c&8)<<3|(c&8)<<4|0xf900), __lsx_vldi((c&1)|(c&1)<<1|(c&2)<<1|(c&2)<<2|(c&4)<<2|(c&4)<<3|(c&8)<<3|(c&8)<<4|0xf900)));
}

XXL_INLINE __m128 lsx_blendv_f32(__m128 a, __m128 b, __m128 c) {
  return (__m128)__lsx_vbitsel_v((__m128i)a, (__m128i)b, __lsx_vslti_w((__m128i)c, 0));
}

XXL_INLINE int lsx_extract_i32(__m128i a, const int b) {
  return (int)__lsx_vpickve2gr_wu(a, (unsigned int)b);
}

XXL_INLINE __m128i lsx_mullo_i32(__m128i a, __m128i b) {
  return __lsx_vmul_w(a, b);
}

XXL_INLINE __m128i lsx_max_i32(__m128i a, __m128i b) {
  return __lsx_vmax_w(a, b);
}

XXL_INLINE __m128i lsx_min_i32(__m128i a, __m128i b) {
  return __lsx_vmin_w(a, b);
}

XXL_INLINE __m128i lsx_cvti8_i32(__m128i a) {
  return __lsx_vsllwil_w_h(__lsx_vsllwil_h_b(a, 0), 0);
}

XXL_INLINE __m128i lsx_shuffle_i8(__m128i a, __m128i b) {
  return __lsx_vand_v(__lsx_vshuf_b(a, a, b), __lsx_vxori_b(__lsx_vslti_b(b, 0), 255));
}

XXL_INLINE __m128i lsx_hadd_i32(__m128i a, __m128i b) {
  return __lsx_vadd_w(__lsx_vpickev_w(b, a), __lsx_vpickod_w(b, a));
}

XXL_INLINE __m128i lsx_maddubs_i16(__m128i a, __m128i b) {
  return __lsx_vsadd_h(__lsx_vmulwev_h_bu_b(a, b), __lsx_vmulwod_h_bu_b(a, b));
}

#endif // __loongarch_sx
