#include <cl_common.h>

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

__kernel void conv2d_1x1_h1w4c1(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter0,
    __read_only image2d_t filter1,
    __read_only image2d_t filter2,
    __read_only image2d_t filter3,
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
  if (out_c >= global_size_dim0 || out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  int out_w1 = out_w + global_size_dim1;
  int out_w2 = mad24(global_size_dim1, 2, out_w);
  int out_w3 = mad24(global_size_dim1, 3, out_w);

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
  int i = 0;
  do {
    CL_DTYPE4 weight0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter0, SAMPLER, (int2)(out_c, i));
    CL_DTYPE4 weight1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter1, SAMPLER, (int2)(out_c, i));
    CL_DTYPE4 weight2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter2, SAMPLER, (int2)(out_c, i));
    CL_DTYPE4 weight3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter3, SAMPLER, (int2)(out_c, i));

    CL_DTYPE4 input0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(mad24(i, input_width, out_w), out_nh));
    CL_DTYPE4 input1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(mad24(i, input_width, out_w1), out_nh));
    CL_DTYPE4 input2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(mad24(i, input_width, out_w2), out_nh));
    CL_DTYPE4 input3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      input_image,
                      SAMPLER,
                      (int2)(mad24(i, input_width, out_w3), out_nh));
    i++;

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
  } while (i < input_c);

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  if (out_w < old_w) {
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(mad24(out_c, old_w, out_w), out_nh));
  }
  if (out_w1 < old_w) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(mad24(out_c, old_w, out_w1), out_nh));
  }
  if (out_w2 < old_w) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(mad24(out_c, old_w, out_w2), out_nh));
  }
  if (out_w3 < old_w) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(mad24(out_c, old_w, out_w3), out_nh));
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

  if (out_w < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(mad24(out_c, old_w, out_w), out_nh),
                          &output0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(mad24(out_c, old_w, out_w), out_nh),
                   output0);
  }

  if (out_w1 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(mad24(out_c, old_w, out_w1), out_nh),
                          &output1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(mad24(out_c, old_w, out_w1), out_nh),
                   output1);
  }

  if (out_w2 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(mad24(out_c, old_w, out_w2), out_nh),
                          &output2);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(mad24(out_c, old_w, out_w2), out_nh),
                   output2);
  }

  if (out_w3 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image,
                          (int2)(mad24(out_c, old_w, out_w3), out_nh),
                          &output3);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output_image,
                   (int2)(mad24(out_c, old_w, out_w3), out_nh),
                   output3);
  }
}
