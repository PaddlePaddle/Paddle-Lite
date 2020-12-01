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

#include <cl_common.h>

__kernel void conv2d_5x5(__private const int global_size_dim0,
                         __private const int global_size_dim1,
                         __private const int global_size_dim2,
                         __read_only image2d_t input_image,
                         __read_only image2d_t filter_image,
                         __read_only image2d_t bias,
#ifdef BATCH_NORM
                         __read_only image2d_t new_scale,
                         __read_only image2d_t new_biase,
#endif
                         __write_only image2d_t output_image,
                         __private const int stride,
                         __private const int offset,
                         __private const int input_c,
                         __private const int dilation,
                         __private const int input_width,  /* of one block */
                         __private const int input_height, /* of one block */
                         __private const int output_width,
                         __private const int output_height) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);

  if (out_c >= global_size_dim0 || out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  const int batch_index = out_nh / output_height;
  const int out_nh_in_one_batch = out_nh % output_height;

  const int filter_n0 = 4 * out_c + 0;
  const int filter_n1 = 4 * out_c + 1;
  const int filter_n2 = 4 * out_c + 2;
  const int filter_n3 = 4 * out_c + 3;

  int2 stride_xy;
  stride_xy.x = stride;
  stride_xy.y = stride;

  int2 ouput_pos_in_one_block;
  ouput_pos_in_one_block.x = out_w;
  ouput_pos_in_one_block.y = out_nh_in_one_batch;

  int2 in_pos_in_one_block;
  in_pos_in_one_block.x = ouput_pos_in_one_block.x * stride + offset;
  in_pos_in_one_block.y = ouput_pos_in_one_block.y * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, output_pos);
#else
  CL_DTYPE4 output = 0.0f;
#endif

  CL_DTYPE4 input;
  CL_DTYPE4 filter[4];
  int2 filter_pos0;
  int2 filter_pos1;
  int2 filter_pos2;
  int2 filter_pos3;
  for (int i = 0; i < input_c; ++i) {
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x,
                         in_pos_in_one_block.y + batch_index * input_height);
    for (int j = 0; j < 5; j++) {
      for (int k = 0; k < 5; k++) {
        input = select(
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input_image,
                          SAMPLER,
                          (int2)(pos_in.x + (j - 2) * dilation,
                                 pos_in.y + (k - 2) * dilation)),
            (CL_DTYPE4)(0.0f),
            (ushort4)(
                (in_pos_in_one_block.x + (j - 2) * dilation < 0 ||
                 in_pos_in_one_block.y + (k - 2) * dilation < 0 ||
                 in_pos_in_one_block.x + (j - 2) * dilation >= input_width ||
                 in_pos_in_one_block.y + (k - 2) * dilation >= input_height)
                << 15));
        int filter_h = k;
        int filter_w = j;
        int filter_c = i;

        filter_pos0.x = filter_c * 5 + filter_w;
        filter_pos0.y = filter_n0 * 5 + filter_h;

        filter_pos1.x = filter_c * 5 + filter_w;
        filter_pos1.y = filter_n1 * 5 + filter_h;

        filter_pos2.x = filter_c * 5 + filter_w;
        filter_pos2.y = filter_n2 * 5 + filter_h;

        filter_pos3.x = filter_c * 5 + filter_w;
        filter_pos3.y = filter_n3 * 5 + filter_h;

        filter[0] =
            READ_IMG_TYPE(CL_DTYPE_CHAR, filter_image, SAMPLER, filter_pos0);
        filter[1] =
            READ_IMG_TYPE(CL_DTYPE_CHAR, filter_image, SAMPLER, filter_pos1);
        filter[2] =
            READ_IMG_TYPE(CL_DTYPE_CHAR, filter_image, SAMPLER, filter_pos2);
        filter[3] =
            READ_IMG_TYPE(CL_DTYPE_CHAR, filter_image, SAMPLER, filter_pos3);

        output.x += dot(input, filter[0]);
        output.y += dot(input, filter[1]);
        output.z += dot(input, filter[2]);
        output.w += dot(input, filter[3]);
        //
        //        if (output_pos.x == 0 && output_pos.y == 5) {
        //          printf("i,j,k ={ %d, %d , %d }\n", i,j,k);
        //          printf("in={ %f , %f , %f , %f } \n",
        //                 convert_float(input.x),
        //                 convert_float(input.y),
        //                 convert_float(input.z),
        //                 convert_float(input.w));
        //          printf("filter0={ %f , %f , %f , %f } \n",
        //                 convert_float(filter[0].x),
        //                 convert_float(filter[0].y),
        //                 convert_float(filter[0].z),
        //                 convert_float(filter[0].w));
        //          printf("filter1={ %f , %f , %f , %f } \n",
        //                 convert_float(filter[1].x),
        //                 convert_float(filter[1].y),
        //                 convert_float(filter[1].z),
        //                 convert_float(filter[1].w));
        //          printf("filter2={ %f , %f , %f , %f } \n",
        //                 convert_float(filter[2].x),
        //                 convert_float(filter[2].y),
        //                 convert_float(filter[2].z),
        //                 convert_float(filter[2].w));
        //          printf("filter3={ %f , %f , %f , %f } \n",
        //                 convert_float(filter[3].x),
        //                 convert_float(filter[3].y),
        //                 convert_float(filter[3].z),
        //                 convert_float(filter[3].w));
        //          printf("output={ %f , %f , %f , %f } \n",
        //                 convert_float(output.x),
        //                 convert_float(output.y),
        //                 convert_float(output.z),
        //                 convert_float(output.w));
        //        }
      }
    }
  }

#ifdef BATCH_NORM
        output =
            output * READ_IMG_TYPE(
                         CL_DTYPE_CHAR, new_scale, SAMPLER, (int2)(out_c, 0)) +
            READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, SAMPLER, (int2)(out_c, 0));
#endif

        output = activation_type4(output);

        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}
