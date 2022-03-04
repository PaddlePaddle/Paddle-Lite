#include <cl_common.h>

__kernel void conv2d_1x1_mali_h1w2c1(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __global CL_DTYPE4 *filter,
    __global CL_DTYPE4 *bias,
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
  if (out_c >= global_size_dim0 || out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  int out_w0 = 2 * out_w;
  int out_w1 = 2 * out_w + 1;

  int out_pos_w0_x = out_c * old_w + out_w0;
  int out_pos_w1_x = out_c * old_w + out_w1;
  int out_pos_y = out_nh;

  int pos_in_w0_x = out_w0 * stride + offset;
  int pos_in_w1_x = out_w1 * stride + offset;
  int pos_in_y = out_nh * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output_w0 = (bias + out_c)[0];
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
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)(out_pos_w0_x, out_pos_y % output_height));
  if (out_w + 1 < output_width) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_pos_w1_x, out_pos_y % output_height));
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
    __global CL_DTYPE4 *bias,
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
  if (get_global_id(0) >= (global_size_dim0 + 1) / 2 ||
      get_global_id(1) >= global_size_dim1 || out_nh >= global_size_dim2) {
    return;
  }

  int pos_in_w0_x = out_w * stride + offset;
  int pos_in_w1_x = (out_w + 1) * stride + offset;
  int pos_in_y = out_nh * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output_w0_c0 = (bias + out_c)[0];
  CL_DTYPE4 output_w1_c0 = output_w0_c0;
  CL_DTYPE4 output_w0_c1 = (bias + out_c + 1)[0];
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
                         (int2)(out_c * old_w + out_w, out_nh % output_height));
  if (out_w + 1 < output_width) {
    alpha1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR,
        prelu_alpha,
        SAMPLER,
        (int2)(out_c * old_w + out_w + 1, out_nh % output_height));
  }
  alpha2 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      prelu_alpha,
      SAMPLER,
      (int2)((out_c + 1) * old_w + out_w, out_nh % output_height));
  if (out_w + 1 < output_width) {
    alpha3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR,
        prelu_alpha,
        SAMPLER,
        (int2)((out_c + 1) * old_w + out_w + 1, out_nh % output_height));
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
    __global CL_DTYPE4 *bias,
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
  if (get_global_id(0) >= (global_size_dim0 + 1) / 2 ||
      get_global_id(1) >= global_size_dim1 ||
      get_global_id(2) >= global_size_dim2) {
    return;
  }

  int pos_in_w0_x = out_w * stride + offset;
  int pos_in_w1_x = (out_w + 1) * stride + offset;
  int pos_in_h0_y = out_nh * stride + offset;
  int pos_in_h1_y = (out_nh + 1) * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 out_w0_h0_c0 = (bias + out_c)[0];
  CL_DTYPE4 out_w1_h0_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w0_h1_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w1_h1_c0 = out_w0_h0_c0;
  CL_DTYPE4 out_w0_h0_c1 = (bias + out_c + 1)[0];
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
                         (int2)(out_c * old_w + out_w, out_nh % output_height));
  if (out_w + 1 < output_width) {
    alpha1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR,
        prelu_alpha,
        SAMPLER,
        (int2)(out_c * old_w + out_w + 1, out_nh % output_height));
  }
  if (out_nh + 1 < output_height) {
    alpha2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR,
        prelu_alpha,
        SAMPLER,
        (int2)(out_c * old_w + out_w, (out_nh + 1) % output_height));
  }
  if (out_w + 1 < output_width && out_nh + 1 < output_height) {
    alpha3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR,
        prelu_alpha,
        SAMPLER,
        (int2)(out_c * old_w + out_w + 1, (out_nh + 1) % output_height));
  }
  alpha4 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      prelu_alpha,
      SAMPLER,
      (int2)((out_c + 1) * old_w + out_w, out_nh % output_height));
  if (out_w + 1 < output_width) {
    alpha5 = READ_IMG_TYPE(
        CL_DTYPE_CHAR,
        prelu_alpha,
        SAMPLER,
        (int2)((out_c + 1) * old_w + out_w + 1, out_nh % output_height));
  }
  if (out_nh + 1 < output_height) {
    alpha6 = READ_IMG_TYPE(
        CL_DTYPE_CHAR,
        prelu_alpha,
        SAMPLER,
        (int2)((out_c + 1) * old_w + out_w, (out_nh + 1) % output_height));
  }
  if (out_w + 1 < output_width && out_nh + 1 < output_height) {
    alpha7 = READ_IMG_TYPE(
        CL_DTYPE_CHAR,
        prelu_alpha,
        SAMPLER,
        (int2)((out_c + 1) * old_w + out_w + 1, (out_nh + 1) % output_height));
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
