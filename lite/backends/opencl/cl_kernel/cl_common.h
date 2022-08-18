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

/////////////////////////////////
// fp16 enabled, MAX_VALUE, MIN_VALUE
/////////////////////////////////
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define MAX_VALUE FLT_MAX
#define MIN_VALUE -FLT_MAX

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

/////////////////////////////////
// CL_DTYPE_float / CL_DTYPE_half
/////////////////////////////////
// Data type: pass one of macros on host: [CL_DTYPE_float, CL_DYPE_half]
#ifdef CL_DTYPE_float
#define CL_DTYPE float
#define CL_DTYPE_CHAR f
#define CL_COMPUTE_DTYPE float
#define CL_COMPUTE_DTYPE_CHAR f
#endif  // CL_DTYPE_float

#ifdef CL_DTYPE_half
#define CL_DTYPE half
#define CL_DTYPE_CHAR h
#ifdef CL_DTYPE_FLOAT_FORCE
#define CL_COMPUTE_DTYPE float
#define CL_COMPUTE_DTYPE_CHAR f
#else
#define CL_COMPUTE_DTYPE half
#define CL_COMPUTE_DTYPE_CHAR h
#endif  // CL_DTYPE_FLOAT_FORCE
#endif  // CL_DTYPE_half

/////////////////////////////////
// GET_VEC_TYPE
/////////////////////////////////
// Note: macro name replacement need twice parser
#define GET_VEC_TYPE(type__, size__) type__##size__
#define VECTORIZED_TYPE(type__, size__) GET_VEC_TYPE(type__, size__)
#define CL_DTYPE4 VECTORIZED_TYPE(CL_DTYPE, 4)
#define CL_DTYPE8 VECTORIZED_TYPE(CL_DTYPE, 8)
#define CL_DTYPE16 VECTORIZED_TYPE(CL_DTYPE, 16)
#define CL_COMPUTE_DTYPE4 VECTORIZED_TYPE(CL_COMPUTE_DTYPE, 4)
#define CL_COMPUTE_DTYPE16 VECTORIZED_TYPE(CL_COMPUTE_DTYPE, 16)

/////////////////////////////////
// CONVERT_TYPE_TO
/////////////////////////////////
#define _CONVERT_TYPE_TO(value, type) convert_##type(value)
#define CONVERT_TYPE_TO(value, type) _CONVERT_TYPE_TO(value, type)

__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

/////////////////////////////////
// WRITE_IMG_TYPE / READ_IMG_TYPE
/////////////////////////////////
#define _WRITE_IMG_TYPE(type_char, img, pos, value) \
  write_image##type_char(img, pos, value)
#define WRITE_IMG_TYPE(type_char, img, pos, value) \
  _WRITE_IMG_TYPE(type_char, img, pos, value)

#define _READ_IMG_TYPE(type_char, img, sampler, pos) \
  read_image##type_char(img, sampler, pos)
#define READ_IMG_TYPE(type_char, img, sampler, pos) \
  _READ_IMG_TYPE(type_char, img, sampler, pos)

/////////////////////////////////
// select macro
// NOTE: a, b must both are vector type
/////////////////////////////////
#ifdef CL_DTYPE_float
#define SELECT(a, b, mask) select(a, b, (uint4)((mask) << 31))
#endif

#ifdef CL_DTYPE_half
#define SELECT(a, b, mask) select(a, b, (ushort4)((mask) << 15))
#endif

/////////////////////////////////
// activation / activation_type4
/////////////////////////////////
inline CL_DTYPE activation(CL_DTYPE in, CL_DTYPE prelu_alpha) {
  CL_DTYPE output = in;
#ifdef PRELU
#ifdef CL_DTYPE_half
  output = select(prelu_alpha * in, in, (ushort)(isgreaterequal(in, 0)));
#else
  output = select(prelu_alpha * in, in, (uint)(isgreaterequal(in, 0)));
#endif
#endif

#ifdef RELU
  output = fmax(in, (CL_DTYPE)0);
#endif

#ifdef RELU6
  output = clamp(in, (CL_DTYPE)0, (CL_DTYPE)6);
#endif

#ifdef LEAKY_RELU
#ifdef CL_DTYPE_half
  output = select(
      (CL_DTYPE)(LEAKY_RELU_ALPHA)*in, in, (ushort)(isgreaterequal(in, 0)));
#else
  output = select(
      (CL_DTYPE)(LEAKY_RELU_ALPHA)*in, in, (uint)(isgreaterequal(in, 0)));
#endif
#endif

#ifdef HARD_SWISH
  output = fmin(fmax(in + (CL_DTYPE)ACT_OFFSET, (CL_DTYPE)0),
                (CL_DTYPE)ACT_THRESHOLD) *
           in / (CL_DTYPE)ACT_SCALE;
#endif

#ifdef HARD_SIGMOID
  output =
      clamp(in * (CL_DTYPE)HARD_SIGMOID_SLOPE + (CL_DTYPE)HARD_SIGMOID_OFFSET,
            (CL_DTYPE)0.0,
            (CL_DTYPE)1.0);
#endif

#ifdef SIGMOID
  output =
      (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * convert_float(in))));
#endif

#ifdef GELU
  const float in_f32 = convert_float(in);
  output = (CL_DTYPE)(0.5f * in_f32 * (1.0f + erf(in_f32 / 1.41421f)));
#endif

#ifdef TANH
  output = (exp(in) - exp(-in)) / (exp(in) + exp(-in));
#endif

#ifdef SWISH
  output = in / (1 + exp(-(CL_DTYPE)ACT_SCALE * in));
#endif

#ifdef EXP
  output = exp(in);
#endif

#ifdef ABS
  output = fabs(in);
#endif

  return output;
}

inline CL_COMPUTE_DTYPE4 activation_type4(CL_COMPUTE_DTYPE4 in,
                                          CL_COMPUTE_DTYPE4 prelu_alpha) {
  CL_COMPUTE_DTYPE4 output = in;
#ifdef PRELU
  output =
      select(prelu_alpha * in, in, isgreaterequal(in, (CL_COMPUTE_DTYPE4)0));
#endif

#ifdef RELU
  output = fmax(in, (CL_COMPUTE_DTYPE4)0);
#endif

#ifdef RELU6
  in = fmax((CL_COMPUTE_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f), in);
  output = fmin((CL_COMPUTE_DTYPE4)(6.0f, 6.0f, 6.0f, 6.0f), in);
#endif

#ifdef LEAKY_RELU
  output = select((CL_COMPUTE_DTYPE4)(LEAKY_RELU_ALPHA)*in,
                  in,
                  isgreaterequal(in, (CL_COMPUTE_DTYPE4)0));
#endif

#ifdef HARD_SWISH
  output = fmin(fmax(in + (CL_COMPUTE_DTYPE4)ACT_OFFSET, (CL_COMPUTE_DTYPE4)0),
                (CL_COMPUTE_DTYPE4)ACT_THRESHOLD) *
           in / (CL_COMPUTE_DTYPE4)ACT_SCALE;
#endif

#ifdef HARD_SIGMOID
  output = clamp(in * (CL_COMPUTE_DTYPE4)HARD_SIGMOID_SLOPE +
                     (CL_COMPUTE_DTYPE4)HARD_SIGMOID_OFFSET,
                 (CL_COMPUTE_DTYPE4)0.0,
                 (CL_COMPUTE_DTYPE4)1.0);
#endif

#ifdef GELU
  const float4 in_f32 = convert_float4(in);
  output.x = (CL_DTYPE)(0.5f * in_f32.x * (1.0f + erf(in_f32.x / 1.41421f)));
  output.y = (CL_DTYPE)(0.5f * in_f32.y * (1.0f + erf(in_f32.y / 1.41421f)));
  output.z = (CL_DTYPE)(0.5f * in_f32.z * (1.0f + erf(in_f32.z / 1.41421f)));
  output.w = (CL_DTYPE)(0.5f * in_f32.w * (1.0f + erf(in_f32.w / 1.41421f)));
#endif

#ifdef SIGMOID
  output.x =
      (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * convert_float(in.x))));
  output.y =
      (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * convert_float(in.y))));
  output.z =
      (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * convert_float(in.z))));
  output.w =
      (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * convert_float(in.w))));
#endif

#ifdef TANH
  output = (exp(in) - exp(-in)) / (exp(in) + exp(-in));
#endif

#ifdef SWISH
  output = in / (1 + exp(-(CL_DTYPE)ACT_SCALE * in));
#endif

#ifdef EXP
  output = exp(in);
#endif

#ifdef ABS
  output = fabs(in);
#endif

  return output;
}

// fuse scale for Elementwise ops
inline CL_COMPUTE_DTYPE4 fuse_scale(CL_COMPUTE_DTYPE4 in,
                                    __private float scale,
                                    __private float bias,
                                    __private float alpha) {
  CL_COMPUTE_DTYPE4 out = CONVERT_TYPE_TO(scale, CL_COMPUTE_DTYPE) * in +
                          CONVERT_TYPE_TO(bias, CL_COMPUTE_DTYPE);
#ifdef FUSE_SCALE_RELU6
  out = clamp(out, (CL_DTYPE4)(0.f), (CL_DTYPE4)(/*alpha=*/6.f));
#endif
  return out;
}

// conv1x1 fuse elementwise_add
inline void elt_fuse_func_wrapper(__read_only image2d_t second_input_image,
                                  const int2 pos,
                                  CL_DTYPE4 *value_p) {
  CL_DTYPE4 second_val =
      READ_IMG_TYPE(CL_DTYPE_CHAR, second_input_image, SAMPLER, pos);
  *value_p += second_val;
#ifdef ELT_ACT_FUSE
  *value_p = fmax(*value_p, (CL_DTYPE4)0);
#endif
}
