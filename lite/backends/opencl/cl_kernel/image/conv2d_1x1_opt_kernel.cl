#include <cl_common.h>

inline elt_fuse_func_wrapper(__read_only image2d_t second_input_image,
                             const int2 pos,
                             CL_DTYPE4 *value_p) {
  CL_DTYPE4 second_val =
      READ_IMG_TYPE(CL_DTYPE_CHAR, second_input_image, SAMPLER, pos);
  *value_p += second_val;
#ifdef ELT_ACT_FUSE
  *value_p = fmax(*value_p, (CL_DTYPE4)0);
#endif
}

__kernel void conv2d_1x1_mali(__read_only image2d_t input,
                              __write_only image2d_t output,
                              __global CL_DTYPE4 *weight,
#ifdef BIASE_CH
                              __global CL_DTYPE4 *bias,
#endif
                              __private const int4 input_shape,
                              __private const int4 output_shape,
                              __private const int2 stride,
                              __private const int4 pad,
                              __read_only image2d_t prelu_alpha) {
  const int out_c_w_idx = get_global_id(0);  // c/4 w/4
  const int out_b_h_idx = get_global_id(1);  // b h

  int N = input_shape.x;
  int IW = input_shape.z;
  int OH = output_shape.y;
  int OW = output_shape.z;
  int CI_SLICES = input_shape.w;
  int CO_SLICES = output_shape.w;
  int OW_SLICES = (output_shape.z + 3) / 4;

  if (out_c_w_idx >= CO_SLICES * OW_SLICES || out_b_h_idx >= N * OH) {
    return;
  }

  const int out_c_blk_idx = out_c_w_idx / OW_SLICES;
  const int out_w_blk_idx = out_c_w_idx % OW_SLICES;
  const int n_idx = out_b_h_idx / OH;
  const int out_h_idx = out_b_h_idx % OH;

  const int out_w4_idx = out_w_blk_idx << 2;

#ifdef BIASE_CH
  CL_DTYPE4 out0 = vload4(out_c_blk_idx, (__global CL_DTYPE *)bias);
  CL_DTYPE4 out1 = out0;
  CL_DTYPE4 out2 = out0;
  CL_DTYPE4 out3 = out0;
#else
  CL_DTYPE4 out0 = 0.0f;
  CL_DTYPE4 out1 = 0.0f;
  CL_DTYPE4 out2 = 0.0f;
  CL_DTYPE4 out3 = 0.0f;
#endif

  CL_DTYPE4 weights0;
  CL_DTYPE4 weights1;
  CL_DTYPE4 weights2;
  CL_DTYPE4 weights3;

  CL_DTYPE4 in0;
  CL_DTYPE4 in1;
  CL_DTYPE4 in2;
  CL_DTYPE4 in3;

  int iw0 = out_w4_idx - pad.z;
  int iw1 = out_w4_idx + 1;
  int iw2 = out_w4_idx + 2;
  int iw3 = out_w4_idx + 3;
  int ih = out_h_idx - pad.x;
  int out_y_idx = mad24(n_idx, input_shape.y, ih);

  iw0 = select(iw0, INT_MIN, iw0 >= IW);
  iw1 = select(iw1, INT_MIN, iw1 >= IW);
  iw2 = select(iw2, INT_MIN, iw2 >= IW);
  iw3 = select(iw3, INT_MIN, iw3 >= IW);

  int in_x_base = 0;
  int weights_offset = mul24(out_c_blk_idx, CI_SLICES << 2);

  for (int ci_slice = 0; ci_slice < CI_SLICES; ++ci_slice) {
    in0 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_x_base + iw0, out_y_idx));
    in1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_x_base + iw1, out_y_idx));
    in2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_x_base + iw2, out_y_idx));
    in3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_x_base + iw3, out_y_idx));

    weights0 = vload4(weights_offset, (__global CL_DTYPE *)weight);
    weights1 = vload4(weights_offset + 1, (__global CL_DTYPE *)weight);
    weights2 = vload4(weights_offset + 2, (__global CL_DTYPE *)weight);
    weights3 = vload4(weights_offset + 3, (__global CL_DTYPE *)weight);

    out0.x += dot(weights0, in0);
    out0.y += dot(weights1, in0);
    out0.z += dot(weights2, in0);
    out0.w += dot(weights3, in0);

    out1.x += dot(weights0, in1);
    out1.y += dot(weights1, in1);
    out1.z += dot(weights2, in1);
    out1.w += dot(weights3, in1);

    out2.x += dot(weights0, in2);
    out2.y += dot(weights1, in2);
    out2.z += dot(weights2, in2);
    out2.w += dot(weights3, in2);

    out3.x += dot(weights0, in3);
    out3.y += dot(weights1, in3);
    out3.z += dot(weights2, in3);
    out3.w += dot(weights3, in3);

    in_x_base += IW;
    weights_offset += 4;
  }

  const int out_x_base = out_c_blk_idx * OW;
  const int remain = OW - out_w4_idx;
  int out_c4w_idx = out_x_base + out_w4_idx;

#if defined(PRELU_CH) || defined(PRELU_ELE) || defined(PRELU_ALL)
  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#endif

#if defined(PRELU_CH)  //{
  alpha0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c_blk_idx, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  if (out_c4w_idx < OW) {
    alpha0 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c4w_idx, out_b_h_idx));
  }
  if ((out_c4w_idx + 1) < OW) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c4w_idx + 1, out_b_h_idx));
  }
  if ((out_c4w_idx + 2) < OW) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c4w_idx + 2, out_b_h_idx));
  }
  if ((out_c4w_idx + 3) < OW) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c4w_idx + 3, out_b_h_idx));
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#endif

#if defined(PRELU_CH) || defined(PRELU_ELE) || defined(PRELU_ALL)
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);
  output2 = activation_type4(output2, alpha2);
  output3 = activation_type4(output3, alpha3);
#else
  out0 = activation_type4(out0, 0.f);
  out1 = activation_type4(out1, 0.f);
  out2 = activation_type4(out2, 0.f);
  out3 = activation_type4(out3, 0.f);
#endif

#ifdef SCALE_ACTIVATION
  out0 = fuse_scale(out0, 1.f, 0.f, 0.f);
  out1 = fuse_scale(out1, 1.f, 0.f, 0.f);
  out2 = fuse_scale(out2, 1.f, 0.f, 0.f);
  out3 = fuse_scale(out3, 1.f, 0.f, 0.f);
#endif

  if (remain >= 4) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx, out_b_h_idx), out0);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx + 1, out_b_h_idx), out1);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx + 2, out_b_h_idx), out2);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx + 3, out_b_h_idx), out3);
  } else if (remain == 3) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx, out_b_h_idx), out0);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx + 1, out_b_h_idx), out1);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx + 2, out_b_h_idx), out2);
  } else if (remain == 2) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx, out_b_h_idx), out0);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx + 1, out_b_h_idx), out1);
  } else if (remain == 1) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(out_c4w_idx, out_b_h_idx), out0);
  }
}

__kernel void conv2d_1x1_opt(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c_block,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

  int outpos_main = mul24(out_c, old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);

  int2 stride_xy = (int2)(stride, stride);

  int2 ouput_pos_in_one_block0 = (int2)(out_w0, out_nh);
  int2 in_pos_in_one_block0 =
      ouput_pos_in_one_block0 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block1 = (int2)(out_w1, out_nh);
  int2 in_pos_in_one_block1 =
      ouput_pos_in_one_block1 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block2 = (int2)(out_w2, out_nh);
  int2 in_pos_in_one_block2 =
      ouput_pos_in_one_block2 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block3 = (int2)(out_w3, out_nh);
  int2 in_pos_in_one_block3 =
      ouput_pos_in_one_block3 * stride_xy + (int2)(offset, offset);

#ifdef BIASE_CH
  CL_DTYPE4 output0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 output1 = output0;
  CL_DTYPE4 output2 = output0;
  CL_DTYPE4 output3 = output0;
#else
  CL_DTYPE4 output0 = 0.0f;
  CL_DTYPE4 output1 = 0.0f;
  CL_DTYPE4 output2 = 0.0f;
  CL_DTYPE4 output3 = 0.0f;
#endif

  int max_w_bound = input_c_block * input_width;
  int burndary_index = input_c_block * 4 - input_c_origin;
  for (int i = 0; i < input_c_block; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x,
                         in_pos_in_one_block0.y);
    CL_DTYPE4 input0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    CL_DTYPE4 weight0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 0));
    CL_DTYPE4 weight1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 1));
    CL_DTYPE4 weight2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 2));
    CL_DTYPE4 weight3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 3));

    if ((max_w_bound - pos_in.x - 1) < input_width &&
        (max_w_bound - pos_in.x - 1) >= 0) {
      if (burndary_index == 0) {
        output0 = mad(input0.x, weight0, output0);
        output0 = mad(input0.y, weight1, output0);
        output0 = mad(input0.z, weight2, output0);
        output0 = mad(input0.w, weight3, output0);
      } else if (burndary_index == 1) {
        output0 = mad(input0.x, weight0, output0);
        output0 = mad(input0.y, weight1, output0);
        output0 = mad(input0.z, weight2, output0);
        output0 = mad(0.0f, weight3, output0);

      } else if (burndary_index == 2) {
        output0 = mad(input0.x, weight0, output0);
        output0 = mad(input0.y, weight1, output0);
        output0 = mad(0.0f, weight2, output0);
        output0 = mad(0.0f, weight3, output0);
      } else if (burndary_index == 3) {
        output0 = mad(input0.x, weight0, output0);
        output0 = mad(0.0f, weight1, output0);
        output0 = mad(0.0f, weight2, output0);
        output0 = mad(0.0f, weight3, output0);
      }
    } else {
      output0 = mad(input0.x, weight0, output0);
      output0 = mad(input0.y, weight1, output0);
      output0 = mad(input0.z, weight2, output0);
      output0 = mad(input0.w, weight3, output0);
    }

    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x,
                    in_pos_in_one_block1.y);
    CL_DTYPE4 input1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    if (abs(max_w_bound - pos_in.x) < input_width) {
      if (burndary_index == 0) {
        output1 = mad(input1.x, weight0, output1);
        output1 = mad(input1.y, weight1, output1);
        output1 = mad(input1.z, weight2, output1);
        output1 = mad(input1.w, weight3, output1);
      } else if (burndary_index == 1) {
        output1 = mad(input1.x, weight0, output1);
        output1 = mad(input1.y, weight1, output1);
        output1 = mad(input1.z, weight2, output1);
        output1 = mad(0.0f, weight3, output1);

      } else if (burndary_index == 2) {
        output1 = mad(input1.x, weight0, output1);
        output1 = mad(input1.y, weight1, output1);
        output1 = mad(0.0f, weight2, output1);
        output1 = mad(0.0f, weight3, output1);
      } else if (burndary_index == 3) {
        output1 = mad(input1.x, weight0, output1);
        output1 = mad(0.0f, weight1, output1);
        output1 = mad(0.0f, weight2, output1);
        output1 = mad(0.0f, weight3, output1);
      }
    } else {
      output1 = mad(input1.x, weight0, output1);
      output1 = mad(input1.y, weight1, output1);
      output1 = mad(input1.z, weight2, output1);
      output1 = mad(input1.w, weight3, output1);
    }

    // -------------2--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x,
                    in_pos_in_one_block2.y);
    CL_DTYPE4 input2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    if (abs(max_w_bound - pos_in.x) < input_width) {
      if (burndary_index == 0) {
        output2 = mad(input2.x, weight0, output2);
        output2 = mad(input2.y, weight1, output2);
        output2 = mad(input2.z, weight2, output2);
        output2 = mad(input2.w, weight3, output2);
      } else if (burndary_index == 1) {
        output2 = mad(input2.x, weight0, output2);
        output2 = mad(input2.y, weight1, output2);
        output2 = mad(input2.z, weight2, output2);
        output2 = mad(0.0f, weight3, output2);

      } else if (burndary_index == 2) {
        output2 = mad(input2.x, weight0, output2);
        output2 = mad(input2.y, weight1, output2);
        output2 = mad(0.0f, weight2, output2);
        output2 = mad(0.0f, weight3, output2);
      } else if (burndary_index == 3) {
        output2 = mad(input2.x, weight0, output2);
        output2 = mad(0.0f, weight1, output2);
        output2 = mad(0.0f, weight2, output2);
        output2 = mad(0.0f, weight3, output2);
      }
    } else {
      output2 = mad(input2.x, weight0, output2);
      output2 = mad(input2.y, weight1, output2);
      output2 = mad(input2.z, weight2, output2);
      output2 = mad(input2.w, weight3, output2);
    }

    // -------------3--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x,
                    in_pos_in_one_block3.y);
    CL_DTYPE4 input3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    if (abs(max_w_bound - pos_in.x) < input_width) {
      if (burndary_index == 0) {
        output3 = mad(input3.x, weight0, output3);
        output3 = mad(input3.y, weight1, output3);
        output3 = mad(input3.z, weight2, output3);
        output3 = mad(input3.w, weight3, output3);
      } else if (burndary_index == 1) {
        output3 = mad(input3.x, weight0, output3);
        output3 = mad(input3.y, weight1, output3);
        output3 = mad(input3.z, weight2, output3);
        output3 = mad(0.0f, weight3, output3);

      } else if (burndary_index == 2) {
        output3 = mad(input3.x, weight0, output3);
        output3 = mad(input3.y, weight1, output3);
        output3 = mad(0.0f, weight2, output3);
        output3 = mad(0.0f, weight3, output3);
      } else if (burndary_index == 3) {
        output3 = mad(input3.x, weight0, output3);
        output3 = mad(0.0f, weight1, output3);
        output3 = mad(0.0f, weight2, output3);
        output3 = mad(0.0f, weight3, output3);
      }
    } else {
      output3 = mad(input3.x, weight0, output3);
      output3 = mad(input3.y, weight1, output3);
      output3 = mad(input3.z, weight2, output3);
      output3 = mad(input3.w, weight3, output3);
    }
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  if (out_w0 < old_w) {
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos0);
  }
  if (out_w1 < old_w) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos1);
  }
  if (out_w2 < old_w) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos2);
  }
  if (out_w3 < old_w) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos3);
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#endif
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);
  output2 = activation_type4(output2, alpha2);
  output3 = activation_type4(output3, alpha3);

#ifdef SCALE_ACTIVATION
  output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
  output1 = fuse_scale(output1, 1.f, 0.f, 0.f);
  output2 = fuse_scale(output2, 1.f, 0.f, 0.f);
  output3 = fuse_scale(output3, 1.f, 0.f, 0.f);
#endif

  if (out_w0 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &output0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos0, output0);
  }

  if (out_w1 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos1, &output1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos1, output1);
  }

  if (out_w2 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos2, &output2);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos2, output2);
  }

  if (out_w3 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos3, &output3);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos3, output3);
  }
}

__kernel void conv2d_1x1_h1w4c1(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

  int outpos_main = mul24(out_c, old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);

  int2 stride_xy = (int2)(stride, stride);

  int2 ouput_pos_in_one_block0 = (int2)(out_w0, out_nh);
  int2 in_pos_in_one_block0 =
      ouput_pos_in_one_block0 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block1 = (int2)(out_w1, out_nh);
  int2 in_pos_in_one_block1 =
      ouput_pos_in_one_block1 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block2 = (int2)(out_w2, out_nh);
  int2 in_pos_in_one_block2 =
      ouput_pos_in_one_block2 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block3 = (int2)(out_w3, out_nh);
  int2 in_pos_in_one_block3 =
      ouput_pos_in_one_block3 * stride_xy + (int2)(offset, offset);

#ifdef BIASE_CH
  CL_DTYPE4 output0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 output1 = output0;
  CL_DTYPE4 output2 = output0;
  CL_DTYPE4 output3 = output0;
#else
  CL_DTYPE4 output0 = 0.0f;
  CL_DTYPE4 output1 = 0.0f;
  CL_DTYPE4 output2 = 0.0f;
  CL_DTYPE4 output3 = 0.0f;
#endif

  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x,
                         in_pos_in_one_block0.y);
    CL_DTYPE4 input0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    CL_DTYPE4 weight0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 0));
    CL_DTYPE4 weight1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 1));
    CL_DTYPE4 weight2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 2));
    CL_DTYPE4 weight3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 3));

    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);

    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x,
                    in_pos_in_one_block1.y);
    CL_DTYPE4 input1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);
    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x,
                    in_pos_in_one_block2.y);
    CL_DTYPE4 input2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);
    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x,
                    in_pos_in_one_block3.y);
    CL_DTYPE4 input3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);
    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  if (out_w0 < old_w) {
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos0);
  }
  if (out_w1 < old_w) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos1);
  }
  if (out_w2 < old_w) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos2);
  }
  if (out_w3 < old_w) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos3);
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#endif
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);
  output2 = activation_type4(output2, alpha2);
  output3 = activation_type4(output3, alpha3);

#ifdef SCALE_ACTIVATION
  output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
  output1 = fuse_scale(output1, 1.f, 0.f, 0.f);
  output2 = fuse_scale(output2, 1.f, 0.f, 0.f);
  output3 = fuse_scale(output3, 1.f, 0.f, 0.f);
#endif

  if (out_w0 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &output0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos0, output0);
  }

  if (out_w1 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos1, &output1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos1, output1);
  }

  if (out_w2 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos2, &output2);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos2, output2);
  }

  if (out_w3 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos3, &output3);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos3, output3);
  }
}

__kernel void conv2d_1x1_h1w2c1(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;

  int outpos_main = mul24(out_c, old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);

  int in_pos_x0 = out_w0 * stride + offset;
  int in_pos_x1 = out_w1 * stride + offset;
  int in_pos_y = out_nh * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 output1 = output0;
#else
  CL_DTYPE4 output0 = 0.0f;
  CL_DTYPE4 output1 = 0.0f;
#endif

  for (int i = 0; i < input_c; ++i) {
    CL_DTYPE4 weight0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2)));
    CL_DTYPE4 weight1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 1));
    CL_DTYPE4 weight2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 2));
    CL_DTYPE4 weight3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 3));

    CL_DTYPE4 input0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x0, in_pos_y));
    CL_DTYPE4 input1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x1, in_pos_y));

    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);

    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);
  }

  CL_DTYPE4 alpha0, alpha1;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  if (out_w0 < old_w) {
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos0);
  }
  if (out_w1 < old_w) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos1);
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
//}
#endif
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);

#ifdef SCALE_ACTIVATION
  output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
  output1 = fuse_scale(output1, 1.f, 0.f, 0.f);
#endif

  if (out_w0 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &output0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos0, output0);
  }

  if (out_w1 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos1, &output1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos1, output1);
  }
}

__kernel void conv2d_1x1_h1w5c1(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + (global_size_dim1 << 1);
  int out_w3 = out_w + global_size_dim1 * 3;
  int out_w4 = out_w + (global_size_dim1 << 2);

  int outpos_main = mul24(out_c, old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);
  int2 output_pos4 = (int2)(outpos_main + out_w4, out_nh);

  int in_pos_x0 = out_w0 * stride + offset;
  int in_pos_x1 = out_w1 * stride + offset;
  int in_pos_x2 = out_w2 * stride + offset;
  int in_pos_x3 = out_w3 * stride + offset;
  int in_pos_x4 = out_w4 * stride + offset;
  int in_pos_y = out_nh * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 output1 = output0;
  CL_DTYPE4 output2 = output0;
  CL_DTYPE4 output3 = output0;
  CL_DTYPE4 output4 = output0;
#else
  CL_DTYPE4 output0 = 0.0f;
  CL_DTYPE4 output1 = 0.0f;
  CL_DTYPE4 output2 = 0.0f;
  CL_DTYPE4 output3 = 0.0f;
  CL_DTYPE4 output4 = 0.0f;
#endif

  for (int i = 0; i < input_c; ++i) {
    CL_DTYPE4 weight0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2)));
    CL_DTYPE4 weight1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 1));
    CL_DTYPE4 weight2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 2));
    CL_DTYPE4 weight3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 3));

    CL_DTYPE4 input0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x0, in_pos_y));
    CL_DTYPE4 input1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x1, in_pos_y));
    CL_DTYPE4 input2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x2, in_pos_y));
    CL_DTYPE4 input3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x3, in_pos_y));
    CL_DTYPE4 input4 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x4, in_pos_y));

    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);

    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);

    output4 = mad(input4.x, weight0, output4);
    output4 = mad(input4.y, weight1, output4);
    output4 = mad(input4.z, weight2, output4);
    output4 = mad(input4.w, weight3, output4);
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3, alpha4;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
  alpha4 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  if (out_w0 < old_w) {
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos0);
  }
  if (out_w1 < old_w) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos1);
  }
  if (out_w2 < old_w) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos2);
  }
  if (out_w3 < old_w) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos3);
  }
  if (out_w4 < old_w) {
    alpha4 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos4);
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
  alpha4 = alpha0;
//}
#endif
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);
  output2 = activation_type4(output2, alpha2);
  output3 = activation_type4(output3, alpha3);
  output4 = activation_type4(output4, alpha4);

#ifdef SCALE_ACTIVATION
  output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
  output1 = fuse_scale(output1, 1.f, 0.f, 0.f);
  output2 = fuse_scale(output2, 1.f, 0.f, 0.f);
  output3 = fuse_scale(output3, 1.f, 0.f, 0.f);
  output4 = fuse_scale(output4, 1.f, 0.f, 0.f);
#endif

  if (out_w0 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &output0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos0, output0);
  }

  if (out_w1 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos1, &output1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos1, output1);
  }

  if (out_w2 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos2, &output2);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos2, output2);
  }

  if (out_w3 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos3, &output3);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos3, output3);
  }

  if (out_w4 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos4, &output4);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos4, output4);
  }
}

__kernel void conv2d_1x1_h1w7c1(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;
  int out_w4 = out_w + global_size_dim1 * 4;
  int out_w5 = out_w + global_size_dim1 * 5;
  int out_w6 = out_w + global_size_dim1 * 6;

  int outpos_main = mul24(out_c, old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);
  int2 output_pos4 = (int2)(outpos_main + out_w4, out_nh);
  int2 output_pos5 = (int2)(outpos_main + out_w5, out_nh);
  int2 output_pos6 = (int2)(outpos_main + out_w6, out_nh);

  int in_pos_x0 = out_w0 * stride + offset;
  int in_pos_x1 = out_w1 * stride + offset;
  int in_pos_x2 = out_w2 * stride + offset;
  int in_pos_x3 = out_w3 * stride + offset;
  int in_pos_x4 = out_w4 * stride + offset;
  int in_pos_x5 = out_w5 * stride + offset;
  int in_pos_x6 = out_w6 * stride + offset;
  int in_pos_y = out_nh * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 output1 = output0;
  CL_DTYPE4 output2 = output0;
  CL_DTYPE4 output3 = output0;
  CL_DTYPE4 output4 = output0;
  CL_DTYPE4 output5 = output0;
  CL_DTYPE4 output6 = output0;
#else
  CL_DTYPE4 output0 = 0.0f;
  CL_DTYPE4 output1 = 0.0f;
  CL_DTYPE4 output2 = 0.0f;
  CL_DTYPE4 output3 = 0.0f;
  CL_DTYPE4 output4 = 0.0f;
  CL_DTYPE4 output5 = 0.0f;
  CL_DTYPE4 output6 = 0.0f;
#endif

  for (int i = 0; i < input_c; ++i) {
    CL_DTYPE4 weight0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2)));
    CL_DTYPE4 weight1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 1));
    CL_DTYPE4 weight2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 2));
    CL_DTYPE4 weight3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 3));

    CL_DTYPE4 input0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x0, in_pos_y));
    CL_DTYPE4 input1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x1, in_pos_y));
    CL_DTYPE4 input2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x2, in_pos_y));
    CL_DTYPE4 input3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x3, in_pos_y));
    CL_DTYPE4 input4 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x4, in_pos_y));
    CL_DTYPE4 input5 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x5, in_pos_y));
    CL_DTYPE4 input6 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_x6, in_pos_y));

    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);

    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);

    output4 = mad(input4.x, weight0, output4);
    output4 = mad(input4.y, weight1, output4);
    output4 = mad(input4.z, weight2, output4);
    output4 = mad(input4.w, weight3, output4);

    output5 = mad(input5.x, weight0, output5);
    output5 = mad(input5.y, weight1, output5);
    output5 = mad(input5.z, weight2, output5);
    output5 = mad(input5.w, weight3, output5);

    output6 = mad(input6.x, weight0, output6);
    output6 = mad(input6.y, weight1, output6);
    output6 = mad(input6.z, weight2, output6);
    output6 = mad(input6.w, weight3, output6);
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
  alpha4 = alpha0;
  alpha5 = alpha0;
  alpha6 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  if (out_w0 < old_w) {
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos0);
  }
  if (out_w1 < old_w) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos1);
  }
  if (out_w2 < old_w) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos2);
  }
  if (out_w3 < old_w) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos3);
  }
  if (out_w4 < old_w) {
    alpha4 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos4);
  }
  if (out_w5 < old_w) {
    alpha5 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos5);
  }
  if (out_w6 < old_w) {
    alpha6 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos6);
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
  alpha4 = alpha0;
  alpha5 = alpha0;
  alpha6 = alpha0;
//}
#endif
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);
  output2 = activation_type4(output2, alpha2);
  output3 = activation_type4(output3, alpha3);
  output4 = activation_type4(output4, alpha4);
  output5 = activation_type4(output5, alpha5);
  output6 = activation_type4(output6, alpha6);

#ifdef SCALE_ACTIVATION
  output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
  output1 = fuse_scale(output1, 1.f, 0.f, 0.f);
  output2 = fuse_scale(output2, 1.f, 0.f, 0.f);
  output3 = fuse_scale(output3, 1.f, 0.f, 0.f);
  output4 = fuse_scale(output4, 1.f, 0.f, 0.f);
  output5 = fuse_scale(output5, 1.f, 0.f, 0.f);
  output6 = fuse_scale(output6, 1.f, 0.f, 0.f);
#endif

  if (out_w0 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &output0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos0, output0);
  }

  if (out_w1 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos1, &output1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos1, output1);
  }

  if (out_w2 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos2, &output2);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos2, output2);
  }

  if (out_w3 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos3, &output3);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos3, output3);
  }

  if (out_w4 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos4, &output4);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos4, output4);
  }

  if (out_w5 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos5, &output5);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos5, output5);
  }

  if (out_w6 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos6, &output6);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos6, output6);
  }
}

__kernel void conv2d_1x1_h2w2c1(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1) * 2;
  const int out_nh = get_global_id(2) * 2;

  int in_pos_w0 = out_w * stride + offset;
  int in_pos_w1 = (out_w + 1) * stride + offset;
  int in_pos_h0 = out_nh * stride + offset;
  int in_pos_h1 = (out_nh + 1) * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 out_w0_h0_c0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 out_w1_h0_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w0_h1_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w1_h1_c0 = out_w0_h0_c0;
#else
  CL_DTYPE4 out_w0_h0_c0 = 0.0f;
  CL_DTYPE4 out_w1_h0_c0 = 0.0f;
  CL_DTYPE4 out_w0_h1_c0 = 0.0f;
  CL_DTYPE4 out_w1_h1_c0 = 0.0f;
#endif
  if (out_w >= output_width || out_nh >= output_height ||
      out_c >= global_size_dim0) {
    return;
  }
  for (int i = 0; i < input_c; ++i) {
    CL_DTYPE4 f0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2)));
    CL_DTYPE4 f1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 1));
    CL_DTYPE4 f2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 2));
    CL_DTYPE4 f3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 3));

    CL_DTYPE4 src_w0_h0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_w0, in_pos_h0));
    CL_DTYPE4 src_w1_h0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_w1, in_pos_h0));
    CL_DTYPE4 src_w0_h1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_w0, in_pos_h1));
    CL_DTYPE4 src_w1_h1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_w1, in_pos_h1));

    out_w0_h0_c0 = mad(f0, src_w0_h0.x, out_w0_h0_c0);
    out_w1_h0_c0 = mad(f0, src_w1_h0.x, out_w1_h0_c0);
    out_w0_h1_c0 = mad(f0, src_w0_h1.x, out_w0_h1_c0);
    out_w1_h1_c0 = mad(f0, src_w1_h1.x, out_w1_h1_c0);
    out_w0_h0_c0 = mad(f1, src_w0_h0.y, out_w0_h0_c0);
    out_w1_h0_c0 = mad(f1, src_w1_h0.y, out_w1_h0_c0);
    out_w0_h1_c0 = mad(f1, src_w0_h1.y, out_w0_h1_c0);
    out_w1_h1_c0 = mad(f1, src_w1_h1.y, out_w1_h1_c0);
    out_w0_h0_c0 = mad(f2, src_w0_h0.z, out_w0_h0_c0);
    out_w1_h0_c0 = mad(f2, src_w1_h0.z, out_w1_h0_c0);
    out_w0_h1_c0 = mad(f2, src_w0_h1.z, out_w0_h1_c0);
    out_w1_h1_c0 = mad(f2, src_w1_h1.z, out_w1_h1_c0);
    out_w0_h0_c0 = mad(f3, src_w0_h0.w, out_w0_h0_c0);
    out_w1_h0_c0 = mad(f3, src_w1_h0.w, out_w1_h0_c0);
    out_w0_h1_c0 = mad(f3, src_w0_h1.w, out_w0_h1_c0);
    out_w1_h1_c0 = mad(f3, src_w1_h1.w, out_w1_h1_c0);
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)(out_c * old_w + out_w, out_nh));
  if (out_w + 1 < output_width) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w + 1, out_nh));
  }
  if (out_nh + 1 < output_height) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w, out_nh + 1));
  }
  if (out_w + 1 < output_width && out_nh + 1 < output_height) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w + 1, out_nh + 1));
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#endif
  out_w0_h0_c0 = activation_type4(out_w0_h0_c0, alpha0);
  out_w1_h0_c0 = activation_type4(out_w1_h0_c0, alpha1);
  out_w0_h1_c0 = activation_type4(out_w0_h1_c0, alpha2);
  out_w1_h1_c0 = activation_type4(out_w1_h1_c0, alpha3);

#ifdef SCALE_ACTIVATION
  out_w0_h0_c0 = fuse_scale(out_w0_h0_c0, 1.f, 0.f, 0.f);
  out_w1_h0_c0 = fuse_scale(out_w1_h0_c0, 1.f, 0.f, 0.f);
  out_w0_h1_c0 = fuse_scale(out_w0_h1_c0, 1.f, 0.f, 0.f);
  out_w1_h1_c0 = fuse_scale(out_w1_h1_c0, 1.f, 0.f, 0.f);
#endif

#ifdef ELT_FUSE
  elt_fuse_func_wrapper(
      second_input_image, (int2)(out_c * old_w + out_w, out_nh), &out_w0_h0_c0);
#endif
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(out_c * old_w + out_w, out_nh),
                 out_w0_h0_c0);
  if (out_w + 1 < output_width) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(out_c * old_w + out_w + 1, out_nh),
                          &out_w1_h0_c0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_c * old_w + out_w + 1, out_nh),
                   out_w1_h0_c0);
  }
  if (out_nh + 1 < output_height) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(out_c * old_w + out_w, out_nh + 1),
                          &out_w0_h1_c0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_c * old_w + out_w, out_nh + 1),
                   out_w0_h1_c0);
  }
  if (out_w + 1 < output_width && out_nh + 1 < output_height) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(out_c * old_w + out_w + 1, out_nh + 1),
                          &out_w1_h1_c0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_c * old_w + out_w + 1, out_nh + 1),
                   out_w1_h1_c0);
  }
}

__kernel void conv2d_1x1_h2w2c2(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0) * 2;
  const int out_w = get_global_id(1) * 2;
  const int out_nh = get_global_id(2) * 2;

  int in_pos_w0 = out_w * stride + offset;
  int in_pos_w1 = (out_w + 1) * stride + offset;
  int in_pos_h0 = out_nh * stride + offset;
  int in_pos_h1 = (out_nh + 1) * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 out_w0_h0_c0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 out_w1_h0_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w0_h1_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w1_h1_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w0_h0_c1 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c + 1, 0));
  CL_DTYPE4 out_w1_h0_c1 = out_w0_h0_c1;
  CL_DTYPE4 out_w0_h1_c1 = out_w0_h0_c1;
  CL_DTYPE4 out_w1_h1_c1 = out_w0_h0_c1;
#else
  CL_DTYPE4 out_w0_h0_c0 = 0.0f;
  CL_DTYPE4 out_w1_h0_c0 = 0.0f;
  CL_DTYPE4 out_w0_h1_c0 = 0.0f;
  CL_DTYPE4 out_w1_h1_c0 = 0.0f;
  CL_DTYPE4 out_w0_h0_c1 = 0.0f;
  CL_DTYPE4 out_w1_h0_c1 = 0.0f;
  CL_DTYPE4 out_w0_h1_c1 = 0.0f;
  CL_DTYPE4 out_w1_h1_c1 = 0.0f;
#endif
  if (out_w >= output_width || out_nh >= output_height ||
      out_c >= global_size_dim0) {
    return;
  }
  for (int i = 0; i < input_c; ++i) {
    CL_DTYPE4 f0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2)));
    CL_DTYPE4 f1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 1));
    CL_DTYPE4 f2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 2));
    CL_DTYPE4 f3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, (i << 2) + 3));
    CL_DTYPE4 f4 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c + 1, (i << 2)));
    CL_DTYPE4 f5 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c + 1, (i << 2) + 1));
    CL_DTYPE4 f6 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c + 1, (i << 2) + 2));
    CL_DTYPE4 f7 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c + 1, (i << 2) + 3));

    CL_DTYPE4 src_w0_h0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_w0, in_pos_h0));
    CL_DTYPE4 src_w1_h0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_w1, in_pos_h0));
    CL_DTYPE4 src_w0_h1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_w0, in_pos_h1));
    CL_DTYPE4 src_w1_h1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(i * input_width + in_pos_w1, in_pos_h1));

    out_w0_h0_c0 += f0 * src_w0_h0.x;
    out_w1_h0_c0 += f0 * src_w1_h0.x;
    out_w0_h1_c0 += f0 * src_w0_h1.x;
    out_w1_h1_c0 += f0 * src_w1_h1.x;
    out_w0_h0_c0 += f1 * src_w0_h0.y;
    out_w1_h0_c0 += f1 * src_w1_h0.y;
    out_w0_h1_c0 += f1 * src_w0_h1.y;
    out_w1_h1_c0 += f1 * src_w1_h1.y;
    out_w0_h0_c0 += f2 * src_w0_h0.z;
    out_w1_h0_c0 += f2 * src_w1_h0.z;
    out_w0_h1_c0 += f2 * src_w0_h1.z;
    out_w1_h1_c0 += f2 * src_w1_h1.z;
    out_w0_h0_c0 += f3 * src_w0_h0.w;
    out_w1_h0_c0 += f3 * src_w1_h0.w;
    out_w0_h1_c0 += f3 * src_w0_h1.w;
    out_w1_h1_c0 += f3 * src_w1_h1.w;
    out_w0_h0_c1 += f4 * src_w0_h0.x;
    out_w1_h0_c1 += f4 * src_w1_h0.x;
    out_w0_h1_c1 += f4 * src_w0_h1.x;
    out_w1_h1_c1 += f4 * src_w1_h1.x;
    out_w0_h0_c1 += f5 * src_w0_h0.y;
    out_w1_h0_c1 += f5 * src_w1_h0.y;
    out_w0_h1_c1 += f5 * src_w0_h1.y;
    out_w1_h1_c1 += f5 * src_w1_h1.y;
    out_w0_h0_c1 += f6 * src_w0_h0.z;
    out_w1_h0_c1 += f6 * src_w1_h0.z;
    out_w0_h1_c1 += f6 * src_w0_h1.z;
    out_w1_h1_c1 += f6 * src_w1_h1.z;
    out_w0_h0_c1 += f7 * src_w0_h0.w;
    out_w1_h0_c1 += f7 * src_w1_h0.w;
    out_w0_h1_c1 += f7 * src_w0_h1.w;
    out_w1_h1_c1 += f7 * src_w1_h1.w;
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
  alpha4 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c + 1, 0));
  alpha5 = alpha4;
  alpha6 = alpha4;
  alpha7 = alpha4;
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)(out_c * old_w + out_w, out_nh));
  if (out_w + 1 < output_width) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w + 1, out_nh));
  }
  if (out_nh + 1 < output_height) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w, out_nh + 1));
  }
  if (out_w + 1 < output_width && out_nh + 1 < output_height) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w + 1, out_nh + 1));
  }
  alpha4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)((out_c + 1) * old_w + out_w, out_nh));
  if (out_w + 1 < output_width) {
    alpha5 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)((out_c + 1) * old_w + out_w + 1, out_nh));
  }
  if (out_nh + 1 < output_height) {
    alpha6 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)((out_c + 1) * old_w + out_w, out_nh + 1));
  }
  if (out_w + 1 < output_width && out_nh + 1 < output_height) {
    alpha7 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)((out_c + 1) * old_w + out_w + 1, out_nh + 1));
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
  alpha4 = alpha0;
  alpha5 = alpha0;
  alpha6 = alpha0;
  alpha7 = alpha0;
//}
#endif
  out_w0_h0_c0 = activation_type4(out_w0_h0_c0, alpha0);
  out_w1_h0_c0 = activation_type4(out_w1_h0_c0, alpha1);
  out_w0_h1_c0 = activation_type4(out_w0_h1_c0, alpha2);
  out_w1_h1_c0 = activation_type4(out_w1_h1_c0, alpha3);
  out_w0_h0_c1 = activation_type4(out_w0_h0_c1, alpha4);
  out_w1_h0_c1 = activation_type4(out_w1_h0_c1, alpha5);
  out_w0_h1_c1 = activation_type4(out_w0_h1_c1, alpha6);
  out_w1_h1_c1 = activation_type4(out_w1_h1_c1, alpha7);

#ifdef SCALE_ACTIVATION
  out_w0_h0_c0 = fuse_scale(out_w0_h0_c0, 1.f, 0.f, 0.f);
  out_w1_h0_c0 = fuse_scale(out_w1_h0_c0, 1.f, 0.f, 0.f);
  out_w0_h1_c0 = fuse_scale(out_w0_h1_c0, 1.f, 0.f, 0.f);
  out_w1_h1_c0 = fuse_scale(out_w1_h1_c0, 1.f, 0.f, 0.f);
  out_w0_h0_c1 = fuse_scale(out_w0_h0_c1, 1.f, 0.f, 0.f);
  out_w1_h0_c1 = fuse_scale(out_w1_h0_c1, 1.f, 0.f, 0.f);
  out_w0_h1_c1 = fuse_scale(out_w0_h1_c1, 1.f, 0.f, 0.f);
  out_w1_h1_c1 = fuse_scale(out_w1_h1_c1, 1.f, 0.f, 0.f);
#endif

  if (out_c >= global_size_dim0) return;
  {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(out_c * old_w + out_w, out_nh),
                          &out_w0_h0_c0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_c * old_w + out_w, out_nh),
                   out_w0_h0_c0);
    if (out_w + 1 < output_width) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)(out_c * old_w + out_w + 1, out_nh),
                            &out_w1_h0_c0);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)(out_c * old_w + out_w + 1, out_nh),
                     out_w1_h0_c0);
    }
    if (out_nh + 1 < output_height) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)(out_c * old_w + out_w, out_nh + 1),
                            &out_w0_h1_c0);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)(out_c * old_w + out_w, out_nh + 1),
                     out_w0_h1_c0);
    }
    if (out_w + 1 < output_width && out_nh + 1 < output_height) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)(out_c * old_w + out_w + 1, out_nh + 1),
                            &out_w1_h1_c0);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)(out_c * old_w + out_w + 1, out_nh + 1),
                     out_w1_h1_c0);
    }
  }
  if (out_c + 1 >= global_size_dim0) return;
  {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)((out_c + 1) * old_w + out_w, out_nh),
                          &out_w0_h0_c1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)((out_c + 1) * old_w + out_w, out_nh),
                   out_w0_h0_c1);
    if (out_w + 1 < output_width) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)((out_c + 1) * old_w + out_w + 1, out_nh),
                            &out_w1_h0_c1);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)((out_c + 1) * old_w + out_w + 1, out_nh),
                     out_w1_h0_c1);
    }
    if (out_nh + 1 < output_height) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)((out_c + 1) * old_w + out_w, out_nh + 1),
                            &out_w0_h1_c1);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)((out_c + 1) * old_w + out_w, out_nh + 1),
                     out_w0_h1_c1);
    }
    if (out_w + 1 < output_width && out_nh + 1 < output_height) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)((out_c + 1) * old_w + out_w + 1, out_nh + 1),
                            &out_w1_h1_c1);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)((out_c + 1) * old_w + out_w + 1, out_nh + 1),
                     out_w1_h1_c1);
    }
  }
}

__kernel void conv2d_1x1_mali_h1w2c1(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __global CL_DTYPE4 *filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = 2 * out_w;
  int out_w1 = 2 * out_w + 1;

  int out_pos_w0_x = out_c * old_w + out_w0;
  int out_pos_w1_x = out_c * old_w + out_w1;
  int out_pos_y = out_nh;

  int pos_in_w0_x = out_w0 * stride + offset;
  int pos_in_w1_x = out_w1 * stride + offset;
  int pos_in_y = out_nh * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output_w0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 output_w1 = output_w0;
#else
  CL_DTYPE4 output_w0 = 0.0f;
  CL_DTYPE4 output_w1 = 0.0f;
#endif

  __global CL_DTYPE4 *weight_ptr = filter + out_c * 4 * input_c;

  for (int i = 0; i < input_c; ++i) {
    CL_DTYPE4 input_w0 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(pos_in_w0_x, pos_in_y));
    CL_DTYPE4 input_w1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(pos_in_w1_x, pos_in_y));

    pos_in_w0_x += input_width;
    pos_in_w1_x += input_width;

    output_w0 += input_w0.x * weight_ptr[0];
    output_w0 += input_w0.y * weight_ptr[1];
    output_w0 += input_w0.z * weight_ptr[2];
    output_w0 += input_w0.w * weight_ptr[3];

    output_w1 += input_w1.x * weight_ptr[0];
    output_w1 += input_w1.y * weight_ptr[1];
    output_w1 += input_w1.z * weight_ptr[2];
    output_w1 += input_w1.w * weight_ptr[3];

    weight_ptr += 4;
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_pos_w0_x, out_pos_y));
  if (out_w + 1 < output_width) {
    alpha1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_pos_w1_x, out_pos_y));
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
//}
#endif
  output_w0 = activation_type4(output_w0, alpha0);
  output_w1 = activation_type4(output_w1, alpha1);

#ifdef SCALE_ACTIVATION
  output_w0 = fuse_scale(output_w0, 1.f, 0.f, 0.f);
  output_w1 = fuse_scale(output_w1, 1.f, 0.f, 0.f);
#endif

  if (out_w0 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(
        second_input_image, (int2)(out_pos_w0_x, out_pos_y), &output_w0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_pos_w0_x, out_pos_y),
                   output_w0);
  }

  if (out_w1 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(
        second_input_image, (int2)(out_pos_w1_x, out_pos_y), &output_w1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_pos_w1_x, out_pos_y),
                   output_w1);
  }
}

__kernel void conv2d_1x1_mali_h1w2c2(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __global CL_DTYPE4 *filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = 2 * get_global_id(0);
  const int out_w = 2 * get_global_id(1);
  const int out_nh = get_global_id(2);

  int pos_in_w0_x = out_w * stride + offset;
  int pos_in_w1_x = (out_w + 1) * stride + offset;
  int pos_in_y = out_nh * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output_w0_c0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 output_w1_c0 = output_w0_c0;
  CL_DTYPE4 output_w0_c1 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c + 1, 0));
  CL_DTYPE4 output_w1_c1 = output_w0_c1;
#else
  CL_DTYPE4 output_w0_c0 = 0.0f;
  CL_DTYPE4 output_w1_c0 = 0.0f;
  CL_DTYPE4 output_w0_c1 = 0.0f;
  CL_DTYPE4 output_w1_c1 = 0.0f;
#endif

  __global CL_DTYPE4 *weight_ptr = filter + out_c * 4 * input_c;

  for (int i = 0; i < input_c; ++i) {
    CL_DTYPE4 input_w0 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(pos_in_w0_x, pos_in_y));
    CL_DTYPE4 input_w1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(pos_in_w1_x, pos_in_y));

    pos_in_w0_x += input_width;
    pos_in_w1_x += input_width;

    output_w0_c0 = mad(input_w0.x, weight_ptr[0], output_w0_c0);
    output_w0_c0 = mad(input_w0.y, weight_ptr[1], output_w0_c0);
    output_w0_c0 = mad(input_w0.z, weight_ptr[2], output_w0_c0);
    output_w0_c0 = mad(input_w0.w, weight_ptr[3], output_w0_c0);

    output_w1_c0 = mad(input_w1.x, weight_ptr[0], output_w1_c0);
    output_w1_c0 = mad(input_w1.y, weight_ptr[1], output_w1_c0);
    output_w1_c0 = mad(input_w1.z, weight_ptr[2], output_w1_c0);
    output_w1_c0 = mad(input_w1.w, weight_ptr[3], output_w1_c0);

    output_w0_c1 = mad(input_w0.x, weight_ptr[4], output_w0_c1);
    output_w0_c1 = mad(input_w0.y, weight_ptr[5], output_w0_c1);
    output_w0_c1 = mad(input_w0.z, weight_ptr[6], output_w0_c1);
    output_w0_c1 = mad(input_w0.w, weight_ptr[7], output_w0_c1);

    output_w1_c1 = mad(input_w1.x, weight_ptr[4], output_w1_c1);
    output_w1_c1 = mad(input_w1.y, weight_ptr[5], output_w1_c1);
    output_w1_c1 = mad(input_w1.z, weight_ptr[6], output_w1_c1);
    output_w1_c1 = mad(input_w1.w, weight_ptr[7], output_w1_c1);

    weight_ptr += 8;
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)(out_c * old_w + out_w, out_nh));
  if (out_w + 1 < output_width) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w + 1, out_nh));
  }
  alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)((out_c + 1) * old_w + out_w, out_nh));
  if (out_w + 1 < output_width) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)((out_c + 1) * old_w + out_w + 1, out_nh));
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#endif
  output_w0_c0 = activation_type4(output_w0_c0, alpha0);
  output_w1_c0 = activation_type4(output_w1_c0, alpha1);
  output_w0_c1 = activation_type4(output_w0_c1, alpha2);
  output_w1_c1 = activation_type4(output_w1_c1, alpha3);

#ifdef SCALE_ACTIVATION
  output_w0_c0 = fuse_scale(output_w0_c0, 1.f, 0.f, 0.f);
  output_w1_c0 = fuse_scale(output_w1_c0, 1.f, 0.f, 0.f);
  output_w0_c1 = fuse_scale(output_w0_c1, 1.f, 0.f, 0.f);
  output_w1_c1 = fuse_scale(output_w1_c1, 1.f, 0.f, 0.f);
#endif
  if (out_c >= global_size_dim0) return;
  {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(out_c * old_w + out_w, out_nh),
                          &output_w0_c0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_c * old_w + out_w, out_nh),
                   output_w0_c0);
    if (out_w + 1 < old_w) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)(out_c * old_w + out_w + 1, out_nh),
                            &output_w1_c0);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)(out_c * old_w + out_w + 1, out_nh),
                     output_w1_c0);
    }
  }
  if (out_c + 1 >= global_size_dim0) return;
  {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)((out_c + 1) * old_w + out_w, out_nh),
                          &output_w0_c1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)((out_c + 1) * old_w + out_w, out_nh),
                   output_w0_c1);
    if (out_w + 1 < old_w) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)((out_c + 1) * old_w + out_w + 1, out_nh),
                            &output_w1_c1);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)((out_c + 1) * old_w + out_w + 1, out_nh),
                     output_w1_c1);
    }
  }
}

__kernel void conv2d_1x1_mali_h2w2c2(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __global CL_DTYPE4 *filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = 2 * get_global_id(0);
  const int out_w = 2 * get_global_id(1);
  const int out_nh = 2 * get_global_id(2);

  int pos_in_w0_x = out_w * stride + offset;
  int pos_in_w1_x = (out_w + 1) * stride + offset;
  int pos_in_h0_y = out_nh * stride + offset;
  int pos_in_h1_y = (out_nh + 1) * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 out_w0_h0_c0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 out_w1_h0_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w0_h1_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w1_h1_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w0_h0_c1 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c + 1, 0));
  CL_DTYPE4 out_w1_h0_c1 = out_w0_h0_c1;
  CL_DTYPE4 out_w0_h1_c1 = out_w0_h0_c1;
  CL_DTYPE4 out_w1_h1_c1 = out_w0_h0_c1;
#else
  CL_DTYPE4 out_w0_h0_c0 = 0.0f;
  CL_DTYPE4 out_w1_h0_c0 = 0.0f;
  CL_DTYPE4 out_w0_h1_c0 = 0.0f;
  CL_DTYPE4 out_w1_h1_c0 = 0.0f;
  CL_DTYPE4 out_w0_h0_c1 = 0.0f;
  CL_DTYPE4 out_w1_h0_c1 = 0.0f;
  CL_DTYPE4 out_w0_h1_c1 = 0.0f;
  CL_DTYPE4 out_w1_h1_c1 = 0.0f;
#endif

  __global CL_DTYPE4 *weight_ptr = filter + out_c * 4 * input_c;

  for (int i = 0; i < input_c; ++i) {
    CL_DTYPE4 input_w0_h0 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(pos_in_w0_x, pos_in_h0_y));
    CL_DTYPE4 input_w0_h1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(pos_in_w0_x, pos_in_h1_y));
    CL_DTYPE4 input_w1_h0 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(pos_in_w1_x, pos_in_h0_y));
    CL_DTYPE4 input_w1_h1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, SAMPLER, (int2)(pos_in_w1_x, pos_in_h1_y));

    pos_in_w0_x += input_width;
    pos_in_w1_x += input_width;

    out_w0_h0_c0 = mad(input_w0_h0.x, weight_ptr[0], out_w0_h0_c0);
    out_w0_h0_c0 = mad(input_w0_h0.y, weight_ptr[1], out_w0_h0_c0);
    out_w0_h0_c0 = mad(input_w0_h0.z, weight_ptr[2], out_w0_h0_c0);
    out_w0_h0_c0 = mad(input_w0_h0.w, weight_ptr[3], out_w0_h0_c0);

    out_w0_h1_c0 = mad(input_w0_h1.x, weight_ptr[0], out_w0_h1_c0);
    out_w0_h1_c0 = mad(input_w0_h1.y, weight_ptr[1], out_w0_h1_c0);
    out_w0_h1_c0 = mad(input_w0_h1.z, weight_ptr[2], out_w0_h1_c0);
    out_w0_h1_c0 = mad(input_w0_h1.w, weight_ptr[3], out_w0_h1_c0);

    out_w1_h0_c0 = mad(input_w1_h0.x, weight_ptr[0], out_w1_h0_c0);
    out_w1_h0_c0 = mad(input_w1_h0.y, weight_ptr[1], out_w1_h0_c0);
    out_w1_h0_c0 = mad(input_w1_h0.z, weight_ptr[2], out_w1_h0_c0);
    out_w1_h0_c0 = mad(input_w1_h0.w, weight_ptr[3], out_w1_h0_c0);

    out_w1_h1_c0 = mad(input_w1_h1.x, weight_ptr[0], out_w1_h1_c0);
    out_w1_h1_c0 = mad(input_w1_h1.y, weight_ptr[1], out_w1_h1_c0);
    out_w1_h1_c0 = mad(input_w1_h1.z, weight_ptr[2], out_w1_h1_c0);
    out_w1_h1_c0 = mad(input_w1_h1.w, weight_ptr[3], out_w1_h1_c0);

    out_w0_h0_c1 = mad(input_w0_h0.x, weight_ptr[4], out_w0_h0_c1);
    out_w0_h0_c1 = mad(input_w0_h0.y, weight_ptr[5], out_w0_h0_c1);
    out_w0_h0_c1 = mad(input_w0_h0.z, weight_ptr[6], out_w0_h0_c1);
    out_w0_h0_c1 = mad(input_w0_h0.w, weight_ptr[7], out_w0_h0_c1);

    out_w0_h1_c1 = mad(input_w0_h1.x, weight_ptr[4], out_w0_h1_c1);
    out_w0_h1_c1 = mad(input_w0_h1.y, weight_ptr[5], out_w0_h1_c1);
    out_w0_h1_c1 = mad(input_w0_h1.z, weight_ptr[6], out_w0_h1_c1);
    out_w0_h1_c1 = mad(input_w0_h1.w, weight_ptr[7], out_w0_h1_c1);

    out_w1_h0_c1 = mad(input_w1_h0.x, weight_ptr[4], out_w1_h0_c1);
    out_w1_h0_c1 = mad(input_w1_h0.y, weight_ptr[5], out_w1_h0_c1);
    out_w1_h0_c1 = mad(input_w1_h0.z, weight_ptr[6], out_w1_h0_c1);
    out_w1_h0_c1 = mad(input_w1_h0.w, weight_ptr[7], out_w1_h0_c1);

    out_w1_h1_c1 = mad(input_w1_h1.x, weight_ptr[4], out_w1_h1_c1);
    out_w1_h1_c1 = mad(input_w1_h1.y, weight_ptr[5], out_w1_h1_c1);
    out_w1_h1_c1 = mad(input_w1_h1.z, weight_ptr[6], out_w1_h1_c1);
    out_w1_h1_c1 = mad(input_w1_h1.w, weight_ptr[7], out_w1_h1_c1);

    weight_ptr += 8;
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
  alpha4 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c + 1, 0));
  alpha5 = alpha4;
  alpha6 = alpha4;
  alpha7 = alpha4;
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)(out_c * old_w + out_w, out_nh));
  if (out_w + 1 < output_width) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w + 1, out_nh));
  }
  if (out_nh + 1 < output_height) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w, out_nh + 1));
  }
  if (out_w + 1 < output_width && out_nh + 1 < output_height) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_c * old_w + out_w + 1, out_nh + 1));
  }
  alpha4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)((out_c + 1) * old_w + out_w, out_nh));
  if (out_w + 1 < output_width) {
    alpha5 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)((out_c + 1) * old_w + out_w + 1, out_nh));
  }
  if (out_nh + 1 < output_height) {
    alpha6 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)((out_c + 1) * old_w + out_w, out_nh + 1));
  }
  if (out_w + 1 < output_width && out_nh + 1 < output_height) {
    alpha7 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)((out_c + 1) * old_w + out_w + 1, out_nh + 1));
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
  alpha4 = alpha0;
  alpha5 = alpha0;
  alpha6 = alpha0;
  alpha7 = alpha0;
//}
#endif
  out_w0_h0_c0 = activation_type4(out_w0_h0_c0, alpha0);
  out_w1_h0_c0 = activation_type4(out_w1_h0_c0, alpha1);
  out_w0_h1_c0 = activation_type4(out_w0_h1_c0, alpha2);
  out_w1_h1_c0 = activation_type4(out_w1_h1_c0, alpha3);
  out_w0_h0_c1 = activation_type4(out_w0_h0_c1, alpha4);
  out_w1_h0_c1 = activation_type4(out_w1_h0_c1, alpha5);
  out_w0_h1_c1 = activation_type4(out_w0_h1_c1, alpha6);
  out_w1_h1_c1 = activation_type4(out_w1_h1_c1, alpha7);

#ifdef SCALE_ACTIVATION
  out_w0_h0_c0 = fuse_scale(out_w0_h0_c0, 1.f, 0.f, 0.f);
  out_w1_h0_c0 = fuse_scale(out_w1_h0_c0, 1.f, 0.f, 0.f);
  out_w0_h1_c0 = fuse_scale(out_w0_h1_c0, 1.f, 0.f, 0.f);
  out_w1_h1_c0 = fuse_scale(out_w1_h1_c0, 1.f, 0.f, 0.f);
  out_w0_h0_c1 = fuse_scale(out_w0_h0_c1, 1.f, 0.f, 0.f);
  out_w1_h0_c1 = fuse_scale(out_w1_h0_c1, 1.f, 0.f, 0.f);
  out_w0_h1_c1 = fuse_scale(out_w0_h1_c1, 1.f, 0.f, 0.f);
  out_w1_h1_c1 = fuse_scale(out_w1_h1_c1, 1.f, 0.f, 0.f);
#endif

  if (out_c >= global_size_dim0) return;
  {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(out_c * old_w + out_w, out_nh),
                          &out_w0_h0_c0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(out_c * old_w + out_w, out_nh),
                   out_w0_h0_c0);
    if (out_w + 1 < output_width) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)(out_c * old_w + out_w + 1, out_nh),
                            &out_w1_h0_c0);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)(out_c * old_w + out_w + 1, out_nh),
                     out_w1_h0_c0);
    }
    if (out_nh + 1 < output_height) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)(out_c * old_w + out_w, out_nh + 1),
                            &out_w0_h1_c0);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)(out_c * old_w + out_w, out_nh + 1),
                     out_w0_h1_c0);
    }
    if (out_w + 1 < output_width && out_nh + 1 < output_height) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)(out_c * old_w + out_w + 1, out_nh + 1),
                            &out_w1_h1_c0);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)(out_c * old_w + out_w + 1, out_nh + 1),
                     out_w1_h1_c0);
    }
  }
  if (out_c + 1 >= global_size_dim0) return;
  {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)((out_c + 1) * old_w + out_w, out_nh),
                          &out_w0_h0_c1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)((out_c + 1) * old_w + out_w, out_nh),
                   out_w0_h0_c1);
    if (out_w + 1 < output_width) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)((out_c + 1) * old_w + out_w + 1, out_nh),
                            &out_w1_h0_c1);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)((out_c + 1) * old_w + out_w + 1, out_nh),
                     out_w1_h0_c1);
    }
    if (out_nh + 1 < output_height) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)((out_c + 1) * old_w + out_w, out_nh + 1),
                            &out_w0_h1_c1);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)((out_c + 1) * old_w + out_w, out_nh + 1),
                     out_w0_h1_c1);
    }
    if (out_w + 1 < output_width && out_nh + 1 < output_height) {
#ifdef ELT_FUSE
      elt_fuse_func_wrapper(second_input_image,
                            (int2)((out_c + 1) * old_w + out_w + 1, out_nh + 1),
                            &out_w1_h1_c1);
#endif
      WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                     output_image,
                     (int2)((out_c + 1) * old_w + out_w + 1, out_nh + 1),
                     out_w1_h1_c1);
    }
  }
}
