#include <cl_common.h>

__kernel void conv2d_1x1_opt(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
#ifdef BATCH_NORM
    __read_only image2d_t new_scale,
    __read_only image2d_t new_biase,
#endif
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
    __private const int old_w) {

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
#elif defined(BIASE_ELE)
  CL_DTYPE4 output0 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, output_pos0);
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

#ifdef BATCH_NORM
  output0 = output0 * READ_IMG_TYPE(
                          CL_DTYPE_CHAR, new_scale, SAMPLER, (int2)(out_c, 0)) +
            READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, SAMPLER, (int2)(out_c, 0));

  output1 = output1 * READ_IMG_TYPE(
                          CL_DTYPE_CHAR, new_scale, SAMPLER, (int2)(out_c, 0)) +
            READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, SAMPLER, (int2)(out_c, 0));

  output2 = output2 * READ_IMG_TYPE(
                          CL_DTYPE_CHAR, new_scale, SAMPLER, (int2)(out_c, 0)) +
            READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, SAMPLER, (int2)(out_c, 0));

  output3 = output3 * READ_IMG_TYPE(
                          CL_DTYPE_CHAR, new_scale, SAMPLER, (int2)(out_c, 0)) +
            READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, SAMPLER, (int2)(out_c, 0));
#endif

  output0 = activation_type4(output0);
  output1 = activation_type4(output1);
  output2 = activation_type4(output2);
  output3 = activation_type4(output3);

  if (out_w0 < old_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos0, output0);
  }

  if (out_w1 < old_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos1, output1);
  }

  if (out_w2 < old_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos2, output2);
  }

  if (out_w3 < old_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos3, output3);
  }
}

__kernel void conv2d_1x1_simple(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
#ifdef BATCH_NORM
    __read_only image2d_t new_scale,
    __read_only image2d_t new_biase,
#endif
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
    __private const int old_w) {
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
#elif defined(BIASE_ELE)
  CL_DTYPE4 output0 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, output_pos0);
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

#ifdef BATCH_NORM
  output0 = output0 * READ_IMG_TYPE(
                          CL_DTYPE_CHAR, new_scale, SAMPLER, (int2)(out_c, 0)) +
            READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, SAMPLER, (int2)(out_c, 0));

  output1 = output1 * READ_IMG_TYPE(
                          CL_DTYPE_CHAR, new_scale, SAMPLER, (int2)(out_c, 0)) +
            READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, SAMPLER, (int2)(out_c, 0));

  output2 = output2 * READ_IMG_TYPE(
                          CL_DTYPE_CHAR, new_scale, SAMPLER, (int2)(out_c, 0)) +
            READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, SAMPLER, (int2)(out_c, 0));

  output3 = output3 * READ_IMG_TYPE(
                          CL_DTYPE_CHAR, new_scale, SAMPLER, (int2)(out_c, 0)) +
            READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, SAMPLER, (int2)(out_c, 0));
#endif

  output0 = activation_type4(output0);
  output1 = activation_type4(output1);
  output2 = activation_type4(output2);
  output3 = activation_type4(output3);

  if (out_w0 < old_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos0, output0);
  }

  if (out_w1 < old_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos1, output1);
  }

  if (out_w2 < old_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos2, output2);
  }

  if (out_w3 < old_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos3, output3);
  }
}
