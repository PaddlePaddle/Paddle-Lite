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
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline half4 activation(half4 in
#ifdef PRELU
,
                        half4 prelu_alpha
#endif
) {
  half4 output;
#ifdef PRELU
  output = select(prelu_alpha * in, in, in >= (half4)0.0);
#endif

#ifdef RELU
  output = fmax(in, (half4)(0.0f));
#endif
  return output;
}

__kernel void feed(__global float *in,
                   __write_only image2d_t output_image,
                   __private const int out_H,
                   __private const int out_W,
                   __private const int out_C,
                   __private const int Stride0,
                   __private const int Stride1,
                   __private const int Stride2) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh/out_H;
  const int out_h = out_nh%out_H;

  const int in_n = out_n;
  const int in_c0 = out_c * 4 + 0;
  const int in_c1 = out_c * 4 + 1;
  const int in_c2 = out_c * 4 + 2;
  const int in_c3 = out_c * 4 + 3;
  const int in_h = out_h;
  const int in_w = out_w;


  int input_pos0 = in_n * Stride2 + in_c0 * Stride1 + in_h * Stride0 + in_w;
  int input_pos1 = in_n * Stride2 + in_c1 * Stride1 + in_h * Stride0 + in_w;
  int input_pos2 = in_n * Stride2 + in_c2 * Stride1 + in_h * Stride0 + in_w;
  int input_pos3 = in_n * Stride2 + in_c3 * Stride1 + in_h * Stride0 + in_w;

  int2 output_pos;
  output_pos.x = out_c * out_W + out_w;
  output_pos.y = out_nh;

  half4 output = (half4)0.0f;
  output.x = convert_half(in[input_pos0]);
  if(out_C - 4 * out_c>=2){
    output.y = convert_half(in[input_pos1]);
  }
  if(out_C - 4 * out_c>=3){
    output.z = convert_half(in[input_pos2]);
  }
  if(out_C - 4 * out_c>=4){
    output.w = convert_half(in[input_pos3]);
  }
  write_imageh(output_image, output_pos, output);

}


__kernel void fetch(__private const int in_height,
                    __private const int in_width,
                    __read_only image2d_t input,
                    __global float* out,
                    __private const int size_ch,
                    __private const int size_block,
                    __private const int size_batch,
                    __private const int C) {
  const int in_c = get_global_id(0);
  const int in_w = get_global_id(1);
  const int in_nh = get_global_id(2);
  const int in_n = in_nh / in_height;
  const int in_h = in_nh % in_height;

  const sampler_t sampler =
          CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  const int pos_x = mad24(in_c, in_width, in_w);
  half4 in = read_imageh(input, sampler, (int2)(pos_x, in_nh));

  const int index = in_n * size_batch + in_c * size_block + in_h * in_width + in_w;
  out[index] = convert_float(in.x);
  if(C - 4 * in_c>=2){
    out[index + size_ch] = convert_float(in.y);
  }
  if(C - 4 * in_c>=3){
    out[index + size_ch * 2] = convert_float(in.z);
  }

  if(C - 4 * in_c>=4){
    out[index + size_ch * 3] = convert_float(in.w);
  }

}


__kernel void scale(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private float scale,
                    __private float bias,
                    __private int out_width) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  int pos_x = mad24(out_c, out_width, out_w);
  half4 in = read_imageh(input, sampler, (int2)(pos_x, out_nh));
  in = convert_half(scale) * in + convert_half(bias);
  write_imageh(output, (int2)(pos_x, out_nh), in);
}

__kernel void sigmoid(__read_only image2d_t input,
                      __write_only image2d_t output) {

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  half4 in = read_imageh(input, sampler, (int2)(x, y));
  half4 out;
  out.x = 1.0 / (1.0 + pow(2.71828182, -1.0 * (float)(in.x)));
  out.y = 1.0 / (1.0 + pow(2.71828182, -1.0 * (float)(in.y)));
  out.z = 1.0 / (1.0 + pow(2.71828182, -1.0 * (float)(in.z)));
  out.w = 1.0 / (1.0 + pow(2.71828182, -1.0 * (float)(in.w)));
  write_imageh(output, (int2)(x, y), out);
}

__kernel void relu6(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private const float threshold) {

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  half4 in = read_imageh(input, sampler, (int2)(x, y));
  in = max((half4)(0.0f, 0.0f, 0.0f, 0.0f), in);
  in = min((half4)(threshold, threshold, threshold, threshold), in);
  write_imageh(output, (int2)(x, y), in);
}

__kernel void relu(__read_only image2d_t input,
                   __write_only image2d_t output){

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  half4 in = read_imageh(input, sampler, (int2)(x, y));
  in = max((half4)(0.0f, 0.0f, 0.0f, 0.0f), in);
  write_imageh(output, (int2)(x, y), in);
}

__kernel void channel_mul(__global image2d_t input, __global image2d_t bias,__write_only
image2d_t outputImage, int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coords;
  coords.x = x;
  coords.y = y;
  int2 coords_bias;
  coords_bias.x = x/w;
  coords_bias.y = 0;
  half4 in = read_imageh(input, sampler, coords);
  half4 biase = read_imageh(bias, sampler, coords_bias);
  half4 output = in * biase;
  write_imageh(outputImage,coords,output);
}


__kernel void reshape(__read_only image2d_t input_image,
                      __write_only image2d_t output_image,
                      __private const int out_C,
                      __private const int out_H,
                      __private const int out_W,
                      __private const int in_W,
                      __private const int in_H,
                      __private const int in_Stride0,
                      __private const int in_Stride1,
                      __private const int in_Stride2,
                      __private const int out_Stride0,
                      __private const int out_Stride1,
                      __private const int out_Stride2) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh/out_H;
  const int out_h = out_nh%out_H;
  const int out_c0 = out_c * 4;
  const int out_c1 = out_c * 4 + 1;
  const int out_c2 = out_c * 4+ 2;
  const int out_c3 = out_c * 4+ 3;

  int count0 =  out_n * out_Stride2 + out_c0 * out_Stride1 + out_h * out_Stride0 + out_w;
  int count1 =  out_n * out_Stride2 + out_c1 * out_Stride1 + out_h * out_Stride0 + out_w;
  int count2 =  out_n * out_Stride2 + out_c2 * out_Stride1 + out_h * out_Stride0 + out_w;
  int count3 =  out_n * out_Stride2 + out_c3 * out_Stride1 + out_h * out_Stride0 + out_w;

  int in_n0 = count0/in_Stride2;
  int in_n1 = count1/in_Stride2;
  int in_n2 = count1/in_Stride2;
  int in_n3 = count2/in_Stride2;

  count0 = count0%in_Stride2;
  count1 = count1%in_Stride2;
  count2 = count2%in_Stride2;
  count3 = count3%in_Stride2;

  int in_c0 = count0/in_Stride1;
  int in_c1 = count1/in_Stride1;
  int in_c2 = count2/in_Stride1;
  int in_c3 = count3/in_Stride1;

  int in_h0 = (count0%in_Stride1)/in_Stride0;
  int in_h1 = (count1%in_Stride1)/in_Stride0;
  int in_h2 = (count2%in_Stride1)/in_Stride0;
  int in_h3 = (count3%in_Stride1)/in_Stride0;

  int in_w0 = (count0%in_Stride1)%in_Stride0;
  int in_w1 = (count1%in_Stride1)%in_Stride0;
  int in_w2 = (count2%in_Stride1)%in_Stride0;
  int in_w3 = (count3%in_Stride1)%in_Stride0;


  int2 input_pos0;
  int2 input_pos1;
  int2 input_pos2;
  int2 input_pos3;

  input_pos0.x = (in_c0/4) * in_W + in_w0;
  input_pos0.y = in_n0 * in_H + in_h0;

  input_pos1.x = (in_c1/4) * in_W + in_w1;
  input_pos1.y = in_n1 * in_H + in_h1;

  input_pos2.x = (in_c2/4) * in_W + in_w2;
  input_pos2.y = in_n2 * in_H + in_h2;

  input_pos3.x = (in_c3/4) * in_W + in_w3;
  input_pos3.y = in_n3 * in_H + in_h3;

  int2 output_pos;
  output_pos.x = out_c * out_W + out_w;
  output_pos.y = out_nh;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP      |
                            CLK_FILTER_NEAREST;

  half4 input0;
  half4 input1;
  half4 input2;
  half4 input3;
  half4 output;

  input0 = read_imageh(input_image, sampler,input_pos0);
  if(in_c0%4==0){
    output.x = input0.x;
  }else if(in_c0%4==1){
    output.x = input0.y;
  }else if(in_c0%4==2){
    output.x = input0.z;
  }else{
    output.x = input0.w;
  }
  if(out_C - out_c * 4>=2){
    input1 = read_imageh(input_image, sampler,input_pos1);
    if(in_c1%4==0){
      output.y = input1.x;
    }else if(in_c1%4==1){
      output.y = input1.y;
    }else if(in_c1%4==2){
      output.y = input1.z;
    }else{
      output.y = input1.w;
    }

  }else{
    output.y = 0.0f;
  }

  if(out_C - out_c * 4>=3){
    input2 = read_imageh(input_image, sampler,input_pos2);

    if(in_c2%4==0){
      output.z = input2.x;
    }else if(in_c2%4==1){
      output.z = input1.y;
    }else if(in_c2%4==2){
      output.z = input2.z;
    }else{
      output.z = input2.w;
    }
  }else{
    output.z = 0.0f;
  }

  if(out_C - out_c * 4>=4){
    input3 = read_imageh(input_image, sampler,input_pos3);
    if(in_c3%4==0){
      output.w = input3.x;
    }else if(in_c3%4==1){
      output.w = input3.y;
    }else if(in_c3%4==2){
      output.w = input3.z;
    }else{
      output.w = input3.w;
    }
  }else{
    output.w = 0.0f;
  }

  write_imageh(output_image, output_pos, output);
}

__kernel void nearest_interp(__read_only image2d_t input, __write_only image2d_t output,
                             __private const float scale_h, __private const float scale_w,
                             __private const int in_dims_h, __private const int out_dims_h,
                             __private const int in_dims_w, __private const int out_dims_w) {
  const int c = get_global_id(0);
  const int w = get_global_id(1);
  const int nh = get_global_id(2);
  int2 output_pos;
  output_pos.x = c * out_dims_w + w;
  output_pos.y = nh;
  int out_n = nh / out_dims_h;
  int out_h = nh % out_dims_h;
  int2 input_pos;
  input_pos.x = c * in_dims_w + w / scale_w;
  input_pos.y = out_n * in_dims_h + out_h / scale_h;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  half4 input_data = read_imageh(input, sampler, (int2)(input_pos.x, input_pos.y));
  write_imageh(output, (int2)(output_pos.x , output_pos.y), input_data);
}

__kernel void elementwise_add(__global image2d_t input, __global image2d_t bias,__write_only image2d_t outputImage) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coords;
  coords.x = x;
  coords.y = y;
  half4 in = read_imageh(input, sampler, coords);
  half4 biase = read_imageh(bias, sampler, coords);
  half4 output = in + biase;
  write_imageh(outputImage,coords,output);
}

__kernel void channel_add(__global image2d_t input, __global image2d_t bias,__write_only image2d_t outputImage,int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coords;
  coords.x = x;
  coords.y = y;
  int2 coords_bias;
  coords_bias.x = x/w;
  coords_bias.y = 0;
  half4 in = read_imageh(input, sampler, coords);
  half4 biase = read_imageh(bias, sampler, coords_bias);
  half4 output = in + biase;
  write_imageh(outputImage,coords,output);
}

__kernel void concatByW(__read_only image2d_t input_image,
                        __write_only image2d_t output_image,
                        __private const int in_W,
                        __private const int pre_Width,
                        __private const int out_Width) {

  const int in_c = get_global_id(0);
  const int in_w = get_global_id(1);
  const int in_nh = get_global_id(2);

  int2 input_pos;
  input_pos.x = in_c * in_W + in_w;
  input_pos.y = in_nh;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  half4 input;
  input = read_imageh(input_image, sampler,input_pos);

  int2 output_pos;
  output_pos.x = input_pos.x + pre_Width + out_Width * in_c;
  output_pos.y = input_pos.y;
  write_imageh(output_image, output_pos, input);

}


__kernel void concatByH(__read_only image2d_t input_image,
                        __write_only image2d_t output_image,
                        __private const int out_W,
                        __private const int out_H_Start) {

  const int in_c = get_global_id(0);
  const int in_w = get_global_id(1);
  const int in_nh = get_global_id(2);

  int2 input_pos;
  input_pos.x = in_c * out_W + in_w;
  input_pos.y = in_nh;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  half4 input;
  input = read_imageh(input_image, sampler,input_pos);

  int2 output_pos;
  output_pos.x = input_pos.x;
  output_pos.y = out_H_Start + input_pos.y;

  write_imageh(output_image, output_pos, input);

}


__kernel void concatByCWith2Inputs(__read_only image2d_t input_image_0,
                                   __read_only image2d_t input_image_1,
                                   __private const int C_0,
                                   __private const int C_1,
                                   __write_only image2d_t output_image,
                                   __private const int out_C,
                                   __private const int out_W) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  int2 output_pos;
  output_pos.x = out_c * out_W + out_w;
  output_pos.y = out_nh;
  half4 output_data;

  for (int i = 0; i < 4; i++) {
    int c = out_c * 4 + i;
    if (c >= out_C) {
      break;
    }
    int c_in;
    half4 input_data;
    if (c < C_0) {
      c_in = c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * out_W + out_w;
      input_pos.y = out_nh;
      input_data = read_imageh(input_image_0, sampler, input_pos);
    } else {
      c_in = c - C_0;
      int2 input_pos;
      input_pos.x = (c_in / 4) * out_W + out_w;
      input_pos.y = out_nh;
      input_data = read_imageh(input_image_1, sampler, input_pos);
    }
    int value_offset = c_in % 4;
    float value;
    if (value_offset == 0) {
      value = input_data.x;
    } else if (value_offset == 1) {
      value = input_data.y;
    } else if (value_offset == 2) {
      value = input_data.z;
    } else if (value_offset == 3) {
      value = input_data.w;
    }
    if (i == 0) {
      output_data.x = value;
    } else if (i == 1) {
      output_data.y = value;
    } else if (i == 2) {
      output_data.z = value;
    } else if (i == 3) {
      output_data.w = value;
    }
  }
  write_imageh(output_image, output_pos, output_data);
}


__kernel void pool_avg(
        __private const int in_height, __private const int in_width,
        __private const int out_height, __private const int out_width,
        __private const int pad_top, __private const int pad_left,
        __private const int stride_h, __private const int stride_w,
        __private const int ksize_h, __private const int ksize_w,
        __read_only image2d_t input, __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  const sampler_t sampler =
          CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int start_h = out_h * stride_h - pad_top;
  int end_h = min(start_h + ksize_h, in_height);
  start_h = max(start_h, 0);

  int start_w = out_w * stride_w - pad_left;
  int end_w = min(start_w + ksize_w, in_width);
  start_w = max(start_w, 0);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  half4 sum = (half4)(0.0f);
  int num = 0 ;
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      sum += read_imageh(input, sampler, (int2)(pos_in_x + x, pos_in_y + y));
    }
  }

  num = ksize_w * ksize_h;
  half4 avg = sum / num;

  const int pos_out_x = mad24(out_c, out_width, out_w);
  write_imageh(output, (int2)(pos_out_x, out_nh), avg);
}


__kernel void depth_conv_3x3s1_DBIASE_CH(__private const int ou_ch_blk,
                               __private const int ou_w_blk,
                               __private const int ou_nh,
                               __read_only image2d_t input,
                               __read_only image2d_t filter,
#if defined(BIASE_CH) || defined(BIASE_ELE)
        __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
__read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif
                               __write_only image2d_t output_image,
                               __private const int stride,
                               __private const int pad,
                               __private const int dilation,
                               __private const int in_ch,
                               __private const int in_w,/* of one block */
                               __private const int in_h, /* of one block */
                               __private const int ou_w,
                               __private const int ou_h) {

  const int ou_ch_blk_id = get_global_id(0);
  const int ou_w_blk_id = get_global_id(1);
  const int ou_nh_id = get_global_id(2);
  const int w_blk_size = 2;

  const int batch_id = ou_nh_id / ou_h;
  int ou_col_id = ou_w_blk_id * w_blk_size;
  int ou_row_id = ou_nh_id % ou_h;
  int ou_x = mad24(ou_ch_blk_id, ou_w, ou_col_id);

  // input pos in one block and on batch
  int col_id = ou_col_id - pad;
  int row_id = ou_row_id - pad;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP          |
                            CLK_FILTER_NEAREST;

#ifdef BIASE_CH
  half4 output[2];
    output[0] = read_imageh(bias, sampler, (int2)(ou_ch_blk_id, 0));
    output[1] = output[0];
#elif defined(BIASE_ELE)
  half4 output[2];
    output[0] = read_imageh(bias, sampler, (int2)(ou_x, ou_nh_id));
    if (ou_col_id + 1 < ou_w) {
        output[1] = read_imageh(bias, sampler, (int2)(ou_x + 1, ou_nh_id));
    }
#else
  half4 output[2] = {0.0f};
#endif

  half4 inputs[12];

  int filter_x = ou_ch_blk_id * 3;
  int filter_y = 0;
  half4 filters[9];
  filters[0] =  read_imageh(filter, sampler,(int2)(filter_x,filter_y));
  filters[1] =  read_imageh(filter, sampler,(int2)(filter_x + 1,filter_y));
  filters[2] =  read_imageh(filter, sampler,(int2)(filter_x + 2,filter_y));

  int in_x = mad24(ou_ch_blk_id, in_w, col_id);
  int in_y = mad24(batch_id, in_h, row_id);

  int y0 = select(in_y, -1, row_id < 0 || row_id >= in_h);
  int x0 = select(in_x, -1, col_id < 0 || col_id >= in_w);
  inputs[0] = read_imageh(input, sampler, (int2)(x0, y0));
  int x1 = select(in_x + 1, -1, col_id + 1 < 0 || col_id + 1 >= in_w);
  inputs[1] = read_imageh(input, sampler, (int2)(x1, y0));
  int x2 = select(in_x + 2, -1, col_id + 2 < 0 || col_id + 2 >= in_w);
  inputs[2] = read_imageh(input, sampler, (int2)(x2, y0));
  int x3 = select(in_x + 3, -1, col_id + 3 < 0 || col_id + 3 >= in_w);
  inputs[3] = read_imageh(input, sampler, (int2)(x3, y0));

  output[0] = mad(inputs[0], filters[0], output[0]);
  output[1] = mad(inputs[1], filters[0], output[1]);

  output[0] = mad(inputs[1], filters[1], output[0]);
  output[1] = mad(inputs[2], filters[1], output[1]);

  output[0] = mad(inputs[2], filters[2], output[0]);
  output[1] = mad(inputs[3], filters[2], output[1]);


  filters[3] =  read_imageh(filter, sampler,(int2)(filter_x,filter_y + 1));
  filters[4] =  read_imageh(filter, sampler,(int2)(filter_x + 1,filter_y + 1));
  filters[5] =  read_imageh(filter, sampler,(int2)(filter_x + 2,filter_y + 1));


  int y1 = select(in_y + 1, -1, row_id + 1 < 0 || row_id + 1 >= in_h);
  inputs[4] = read_imageh(input, sampler, (int2)(x0, y1));
  inputs[5] = read_imageh(input, sampler, (int2)(x1, y1));
  inputs[6] = read_imageh(input, sampler, (int2)(x2, y1));
  inputs[7] = read_imageh(input, sampler, (int2)(x3, y1));


  output[0] = mad(inputs[4], filters[3], output[0]);
  output[1] = mad(inputs[5], filters[3], output[1]);

  output[0] = mad(inputs[5], filters[4], output[0]);
  output[1] = mad(inputs[6], filters[4], output[1]);

  output[0] = mad(inputs[6], filters[5], output[0]);
  output[1] = mad(inputs[7], filters[5], output[1]);


  filters[6] =  read_imageh(filter, sampler,(int2)(filter_x,filter_y + 2));
  filters[7] =  read_imageh(filter, sampler,(int2)(filter_x + 1,filter_y + 2));
  filters[8] =  read_imageh(filter, sampler,(int2)(filter_x + 2,filter_y + 2));

  int y2 = select(in_y + 2, -1, row_id + 2 < 0 || row_id + 2 >= in_h);
  inputs[8] = read_imageh(input, sampler, (int2)(x0, y2));
  inputs[9] = read_imageh(input, sampler, (int2)(x1, y2));
  inputs[10] = read_imageh(input, sampler, (int2)(x2, y2));
  inputs[11] = read_imageh(input, sampler, (int2)(x3, y2));


  output[0] = mad(inputs[8], filters[6], output[0]);
  output[1] = mad(inputs[9], filters[6], output[1]);

  output[0] = mad(inputs[9], filters[7], output[0]);
  output[1] = mad(inputs[10], filters[7], output[1]);

  output[0] = mad(inputs[10], filters[8], output[0]);
  output[1] = mad(inputs[11], filters[8], output[1]);
#ifdef BATCH_NORM
  half4 scale = read_imageh(new_scale, sampler, (int2)(ou_ch_blk_id, 0));
    half4 biase = read_imageh(new_biase, sampler, (int2)(ou_ch_blk_id, 0));
    output[0] = mad(scale, output[0], biase);
    if (ou_col_id + 1 < ou_w) {
        output[1] = mad(scale, output[1], biase);
    }
#endif

  write_imageh(output_image, (int2)(ou_x, ou_nh_id), output[0]);
  if (ou_col_id + 1 < ou_w) {
    write_imageh(output_image, (int2)(ou_x + 1, ou_nh_id), output[1]);
  }

}


__kernel void depth_conv_3x3_DBIASE_CH(__private const int global_size_dim0,
                             __private const int global_size_dim1,
                             __private const int global_size_dim2,
                             __read_only image2d_t input,
                             __read_only image2d_t filter,
#if defined(BIASE_CH) || defined(BIASE_ELE)
        __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
__read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif
                             __write_only image2d_t output_image,
                             __private const int stride,
                             __private const int offset,
                             __private const int input_c,
                             __private const int dilation,
                             __private const int input_width,/* of one block */
                             __private const int input_height, /* of one block */
                             __private const int output_width,
                             __private const int output_height) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);


  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP          |
                            CLK_FILTER_NEAREST;

  const int batch_index = out_nh / output_height;

  const int out_nh_in_one_batch = out_nh % output_height;


  int2 stride_xy = (int2)(stride, stride);
  int2 ouput_pos_in_one_block = (int2)(out_w, out_nh_in_one_batch);

  int2 in_pos_in_one_block = ouput_pos_in_one_block * stride_xy + (int2)(offset, offset);

#ifdef BIASE_CH
  half4 output = read_imageh(bias, sampler, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  half4 output = read_imageh(bias, sampler, output_pos);
#else
  half4 output = 0.0f;
#endif

  const int filter_width = 3;
  const int filter_height = 3;

  int2 pos_in_input_block = (int2)(out_c * input_width, batch_index * input_height);

  int2 pos_in_filter_block = (int2)(out_c * filter_width, batch_index * filter_height);

  int filter_x = pos_in_filter_block.x ;
  int filter_y = pos_in_filter_block.y ;

  half4 inputs[9];

  inputs[0] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                     (half4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y - 1 >= input_height) << 15));

  inputs[1] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                     (half4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y - 1 >= input_height) << 15));

  inputs[2] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                     (half4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y - 1 >= input_height) << 15));

  inputs[3] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y)),
                     (half4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y >= input_height) << 15));
  /*
  if (output_pos.x == 112 && output_pos.y == 0) {
        half4 input1 = inputs[3];
        float4 in = (float4)(input1.x, input1.y, input1.z, input1.w);
        printf(" input4 3 - %v4hlf \n", in);
        printf(" --- %d ---\n", in_pos_in_one_block.x - 1);
  }
  */


  inputs[4] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y)),
                     (half4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y >= input_height) << 15));

  inputs[5] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y)),
                     (half4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y >= input_height) << 15));

  inputs[6] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                     (half4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y + 1 >= input_height) << 15));

  inputs[7] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                     (half4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y + 1 >= input_height) << 15));

  inputs[8] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                     (half4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y + 1 >= input_height) << 15));

  half4 filters[9];
  filters[0] =  read_imageh(filter, sampler,(int2)(filter_x,filter_y));
  filters[1] =  read_imageh(filter, sampler,(int2)(filter_x + 1,filter_y));
  filters[2] =  read_imageh(filter, sampler,(int2)(filter_x + 2,filter_y));
  filters[3] =  read_imageh(filter, sampler,(int2)(filter_x,filter_y + 1));
  filters[4] =  read_imageh(filter, sampler,(int2)(filter_x + 1,filter_y + 1));
  filters[5] =  read_imageh(filter, sampler,(int2)(filter_x + 2,filter_y + 1));
  filters[6] =  read_imageh(filter, sampler,(int2)(filter_x,filter_y + 2));
  filters[7] =  read_imageh(filter, sampler,(int2)(filter_x + 1,filter_y + 2));
  filters[8] =  read_imageh(filter, sampler,(int2)(filter_x + 2,filter_y + 2));

  for(int i = 0 ;i < 9 ; i++){
    output += inputs[i] * filters[i];
  }
#ifdef BATCH_NORM
  output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

  /*

  if (output_pos.x == 112 && output_pos.y == 0) {

      for (int i = 0; i < 9; ++i) {
          half4 input1 = inputs[i];
          float4 in = (float4)(input1.x, input1.y, input1.z, input1.w);
          printf(" input4 %d - %v4hlf \n", i, in);
      }

      float4 out = (float4)(output.x, output.y, output.z, output.w);
      printf(" depth wise output output4 = %v4hlf \n", out);
      printf(" pos_in_input_block -x %d \n ", pos_in_input_block.x);
      printf(" pos_in_input_block -y %d \n ", pos_in_input_block.y);
      printf(" in_pos_in_one_block - x %d \n", in_pos_in_one_block.x);
      printf(" in_pos_in_one_block - y %d \n", in_pos_in_one_block.y);
  }

  */

  write_imageh(output_image, output_pos, output);

}

__kernel void conv_1x1_spl_DRELU_DBIASE_CH(
        __private const int global_size_dim0, __private const int global_size_dim1,
        __private const int global_size_dim2, __read_only image2d_t input_image,
        __read_only image2d_t filter,
#if defined(BIASE_CH) || defined(BIASE_ELE)
        __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
        __read_only image2d_t new_scale, __read_only image2d_t new_biase,
#endif
        __write_only image2d_t output_image, __private const int stride,
        __private const int offset, __private const int input_c,__private const int input_c_origin,
        __private const int dilation,
        __private const int input_width,  /* of one block */
        __private const int input_height, /* of one block */
        __private const int output_width,
        __private const int output_height,
        __private const int old_w
) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

  int outpos_main = mul24(out_c , old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);

  const sampler_t sampler =
          CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

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
  half4 output0 = read_imageh(bias, sampler, (int2)(out_c, 0));
    half4 output1 = read_imageh(bias, sampler, (int2)(out_c, 0));
    half4 output2 = read_imageh(bias, sampler, (int2)(out_c, 0));
    half4 output3 = read_imageh(bias, sampler, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  half4 output0 = read_imageh(bias, sampler, output_pos0);
    half4 output1 = read_imageh(bias, sampler, output_pos1);
    half4 output2 = read_imageh(bias, sampler, output_pos2);
    half4 output3 = read_imageh(bias, sampler, output_pos3);

#else
  half4 output0 = 0.0f;
  half4 output1 = 0.0f;
  half4 output2 = 0.0f;
  half4 output3 = 0.0f;
#endif

  int max_w_bound = input_c * input_width;
  int burndary_index = input_c * 4 - input_c_origin;
  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x, in_pos_in_one_block0.y);
    half4 input0 = read_imageh(input_image, sampler, pos_in);

    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
    half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
    half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
    half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));

    int bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input0.w = 0.0f;
      } else if (burndary_index==2){
        input0.z = 0.0f;
        input0.w = 0.0f;
      } else if (burndary_index==3){
        input0.y = 0.0f;
        input0.z = 0.0f;
        input0.w = 0.0f;
      }
    }
    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);
    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x, in_pos_in_one_block1.y);
    half4 input1 = read_imageh(input_image, sampler, pos_in);

    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input1.w = 0.0f;
      } else if (burndary_index==2){
        input1.z = 0.0f;
        input1.w = 0.0f;
      } else if (burndary_index==3){
        input1.y = 0.0f;
        input1.z = 0.0f;
        input1.w = 0.0f;
      }
    }
    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    // -------------2--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x, in_pos_in_one_block2.y);
    half4 input2 = read_imageh(input_image, sampler, pos_in);

    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input2.w = 0.0f;
      } else if (burndary_index==2){
        input2.z = 0.0f;
        input2.w = 0.0f;
      } else if (burndary_index==3){
        input2.y = 0.0f;
        input2.z = 0.0f;
        input2.w = 0.0f;
      }
    }
    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    // -------------3--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x, in_pos_in_one_block3.y);
    half4 input3 = read_imageh(input_image, sampler, pos_in);
    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input3.w = 0.0f;
      } else if (burndary_index==2){
        input3.z = 0.0f;
        input3.w = 0.0f;
      } else if (burndary_index==3){
        input3.y = 0.0f;
        input3.z = 0.0f;
        input3.w = 0.0f;
      }
    }

    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);
  }

#ifdef BATCH_NORM
  output0 = output0 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output1 = output1 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output2 = output2 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output3 = output3 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef RELU
  output0 = activation(output0);
     output1 = activation(output1);
     output2 = activation(output2);
     output3 = activation(output3);
#endif

  if (out_w0 < old_w) {
    write_imageh(output_image, output_pos0, output0);
  }

  if (out_w1 < old_w){
    write_imageh(output_image, output_pos1, output1);
  }

  if (out_w2 < old_w){
    write_imageh(output_image, output_pos2, output2);
  }

  if (out_w3 < old_w){
    write_imageh(output_image, output_pos3, output3);
  }
}


__kernel void conv_3x3_DRELU_DBIASE_CH(__private const int global_size_dim0,
                       __private const int global_size_dim1,
                       __private const int global_size_dim2,
                       __read_only image2d_t input_image,
                       __read_only image2d_t filter,

#if defined(BIASE_CH) || defined(BIASE_ELE)
        __read_only image2d_t bias,
#endif

#ifdef BATCH_NORM
__read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif

                       __write_only image2d_t output_image,
                       __private const int stride,
                       __private const int offset,
                       __private const int input_c,
                       __private const int dilation,
                       __private const int input_width,/* of one block */
                       __private const int input_height,/* of one block */
                       __private const int output_width,
                       __private const int output_height,
                       __private const int output_c,
                       __private const int filter_channel,
                       __private const int group) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);

  if (out_c >= global_size_dim0 ||
      out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }


  int2 stride_xy;
  stride_xy.x = stride;
  stride_xy.y = stride;

  int2 ouput_pos_in_one_block;
  ouput_pos_in_one_block.x = out_w;
  ouput_pos_in_one_block.y = out_nh;


  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP          |
                            CLK_FILTER_NEAREST;

  int2 in_pos_in_one_block;
  in_pos_in_one_block.x = ouput_pos_in_one_block.x * stride + offset;
  in_pos_in_one_block.y = ouput_pos_in_one_block.y * stride + offset;

#ifdef BIASE_CH
  half4 output = read_imageh(bias, sampler, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  half4 output = read_imageh(bias, sampler, output_pos);
#else
  half4 output = 0.0f;
#endif

  half4 input[9];
  if (group == 1) {
    for (int i = 0; i < input_c; ++i) {
      int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
      input[0] = select(read_imageh(input_image, sampler,
                                    (int2)(pos_in.x - dilation, pos_in.y - dilation)),
                        (half4)(0.0f),
                        (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

      input[1] = select(read_imageh(input_image, sampler,
                                    (int2)(pos_in.x, pos_in.y - dilation)),
                        (half4)(0.0f),
                        (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

      input[2] = select(read_imageh(input_image, sampler,
                                    (int2)(pos_in.x + dilation, pos_in.y - dilation)),
                        (half4)(0.0f),
                        (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

      input[3] = select(read_imageh(input_image, sampler,
                                    (int2)(pos_in.x - dilation, pos_in.y)),
                        (half4)(0.0f),
                        (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));

      input[4] = select(read_imageh(input_image, sampler,
                                    (int2)(pos_in.x, pos_in.y)),
                        (half4)(0.0f),
                        (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y >= input_height) << 15));

      input[5] = select(read_imageh(input_image, sampler,
                                    (int2)(pos_in.x + dilation, pos_in.y)),
                        (half4)(0.0f),
                        (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));

      input[6] = select(read_imageh(input_image, sampler,
                                    (int2)(pos_in.x - dilation, pos_in.y + dilation)),
                        (half4)(0.0f),
                        (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

      input[7] = select(read_imageh(input_image, sampler,
                                    (int2)(pos_in.x, pos_in.y + dilation)),
                        (half4)(0.0f),
                        (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

      input[8] = select(read_imageh(input_image, sampler,
                                    (int2)(pos_in.x + dilation, pos_in.y + dilation)),
                        (half4)(0.0f),
                        (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));


/*
            for (int j = 0; j < 9; ++j) {
                int2 pos_of_weight;
                pos_of_weight.x = i * 3 + j % 3;
                pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
                half4 weight_x = read_imageh(filter, sampler, pos_of_weight);
                output.x += dot(input[j], weight_x);

                pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
                half4 weight_y = read_imageh(filter, sampler, pos_of_weight);
                output.y += dot(input[j], weight_y);

                pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
                half4 weight_z = read_imageh(filter, sampler, pos_of_weight);
                output.z += dot(input[j], weight_z);

                pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
                half4 weight_w = read_imageh(filter, sampler, pos_of_weight);
                output.w += dot(input[j], weight_w);
            }
*/
      int j = 0;
      int2 pos_of_weight;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      half4 weight_x = read_imageh(filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      half4 weight_y = read_imageh(filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      half4 weight_z = read_imageh(filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      half4 weight_w = read_imageh(filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 1;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = read_imageh(filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = read_imageh(filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = read_imageh(filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = read_imageh(filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 2;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = read_imageh(filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = read_imageh(filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = read_imageh(filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = read_imageh(filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 3;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = read_imageh(filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = read_imageh(filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = read_imageh(filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = read_imageh(filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 4;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = read_imageh(filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = read_imageh(filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = read_imageh(filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = read_imageh(filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 5;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = read_imageh(filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = read_imageh(filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = read_imageh(filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = read_imageh(filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 6;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = read_imageh(filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = read_imageh(filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = read_imageh(filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = read_imageh(filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 7;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = read_imageh(filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = read_imageh(filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = read_imageh(filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = read_imageh(filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 8;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = read_imageh(filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = read_imageh(filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = read_imageh(filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = read_imageh(filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

    }
  } else {
    for (int i = 0; i < 4; i++) {
      int used_input_channel_num = (out_c * 4 + i) / (output_c / group) * filter_channel;
      for (int f_c = 0; f_c < filter_channel; ++f_c) {
        int input_c = used_input_channel_num + f_c;
        int input_block = input_c / 4;
        int2 pos_in = (int2)(input_block * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
        input[0] = select(read_imageh(input_image, sampler,
                                      (int2)(pos_in.x - dilation, pos_in.y - dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));
        input[1] = select(read_imageh(input_image, sampler,
                                      (int2)(pos_in.x, pos_in.y - dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));
        input[2] = select(read_imageh(input_image, sampler,
                                      (int2)(pos_in.x + dilation, pos_in.y - dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));
        input[3] = select(read_imageh(input_image, sampler,
                                      (int2)(pos_in.x - dilation, pos_in.y)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));
        input[4] = select(read_imageh(input_image, sampler,
                                      (int2)(pos_in.x, pos_in.y)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y >= input_height) << 15));
        input[5] = select(read_imageh(input_image, sampler,
                                      (int2)(pos_in.x + dilation, pos_in.y)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));
        input[6] = select(read_imageh(input_image, sampler,
                                      (int2)(pos_in.x - dilation, pos_in.y + dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));
        input[7] = select(read_imageh(input_image, sampler,
                                      (int2)(pos_in.x, pos_in.y + dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));
        input[8] = select(read_imageh(input_image, sampler,
                                      (int2)(pos_in.x + dilation, pos_in.y + dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

        half tmp_out = 0;
        for (int j = 0; j < 9; j++) {
          int2 pos_of_weight;
          pos_of_weight.x = (f_c / 4) * 3 + j % 3;
          pos_of_weight.y = out_c * 4 * 3 + i * 3 + j / 3;
          half4 weight = read_imageh(filter, sampler, pos_of_weight);
          int f_c_offset = f_c % 4;
          half f_value;
          if (f_c_offset == 0) {
            f_value = weight.x;
          } else if (f_c_offset == 1) {
            f_value = weight.y;
          } else if (f_c_offset == 2) {
            f_value = weight.z;
          } else if (f_c_offset == 3) {
            f_value = weight.w;
          }
          int input_c_offset = input_c % 4;
          half input_value;
          if (input_c_offset == 0) {
            input_value = input[j].x;
          } else if (input_c_offset == 1) {
            input_value = input[j].y;
          } else if (input_c_offset == 2) {
            input_value = input[j].z;
          } else if (input_c_offset == 3) {
            input_value = input[j].w;
          }
          tmp_out += f_value * input_value;
        }

        if (i == 0) {
          output.x += tmp_out;
        } else if (i == 1) {
          output.y += tmp_out;
        } else if (i == 2) {
          output.z += tmp_out;
        } else if (i == 3) {
          output.w += tmp_out;
        }
      }
    }
  }


#ifdef BATCH_NORM
  output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef RELU
  output = activation(output);
#endif

  write_imageh(output_image, output_pos, output);
}


__kernel void conv_1x1_spl(
        __private const int global_size_dim0, __private const int global_size_dim1,
        __private const int global_size_dim2, __read_only image2d_t input_image,
        __read_only image2d_t filter,
        __write_only image2d_t output_image, __private const int stride,
        __private const int offset, __private const int input_c,__private const int input_c_origin,
        __private const int dilation,
        __private const int input_width,  /* of one block */
        __private const int input_height, /* of one block */
        __private const int output_width,
        __private const int output_height,
        __private const int old_w
) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

  int outpos_main = mul24(out_c , old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);

  const sampler_t sampler =
          CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

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

  half4 output0 = 0.0f;
  half4 output1 = 0.0f;
  half4 output2 = 0.0f;
  half4 output3 = 0.0f;

  int max_w_bound = input_c * input_width;
  int burndary_index = input_c * 4 - input_c_origin;
  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x, in_pos_in_one_block0.y);
    half4 input0 = read_imageh(input_image, sampler, pos_in);

    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
    half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
    half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
    half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));

    int bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input0.w = 0.0f;
      } else if (burndary_index==2){
        input0.z = 0.0f;
        input0.w = 0.0f;
      } else if (burndary_index==3){
        input0.y = 0.0f;
        input0.z = 0.0f;
        input0.w = 0.0f;
      }
    }
    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);
    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x, in_pos_in_one_block1.y);
    half4 input1 = read_imageh(input_image, sampler, pos_in);

    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input1.w = 0.0f;
      } else if (burndary_index==2){
        input1.z = 0.0f;
        input1.w = 0.0f;
      } else if (burndary_index==3){
        input1.y = 0.0f;
        input1.z = 0.0f;
        input1.w = 0.0f;
      }
    }
    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    // -------------2--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x, in_pos_in_one_block2.y);
    half4 input2 = read_imageh(input_image, sampler, pos_in);

    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input2.w = 0.0f;
      } else if (burndary_index==2){
        input2.z = 0.0f;
        input2.w = 0.0f;
      } else if (burndary_index==3){
        input2.y = 0.0f;
        input2.z = 0.0f;
        input2.w = 0.0f;
      }
    }
    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    // -------------3--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x, in_pos_in_one_block3.y);
    half4 input3 = read_imageh(input_image, sampler, pos_in);
    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input3.w = 0.0f;
      } else if (burndary_index==2){
        input3.z = 0.0f;
        input3.w = 0.0f;
      } else if (burndary_index==3){
        input3.y = 0.0f;
        input3.z = 0.0f;
        input3.w = 0.0f;
      }
    }

    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);
  }

  if (out_w0 < old_w) {
    write_imageh(output_image, output_pos0, output0);
  }

  if (out_w1 < old_w){
    write_imageh(output_image, output_pos1, output1);
  }

  if (out_w2 < old_w){
    write_imageh(output_image, output_pos2, output2);
  }

  if (out_w3 < old_w){
    write_imageh(output_image, output_pos3, output3);
  }
}

__kernel void conv_1x1_spl_DBIASE_CH(
        __private const int global_size_dim0, __private const int global_size_dim1,
        __private const int global_size_dim2, __read_only image2d_t input_image,
        __read_only image2d_t filter,
#if defined(BIASE_CH) || defined(BIASE_ELE)
        __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
        __read_only image2d_t new_scale, __read_only image2d_t new_biase,
#endif
        __write_only image2d_t output_image, __private const int stride,
        __private const int offset, __private const int input_c,__private const int input_c_origin,
        __private const int dilation,
        __private const int input_width,  /* of one block */
        __private const int input_height, /* of one block */
        __private const int output_width,
        __private const int output_height,
        __private const int old_w
) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

  int outpos_main = mul24(out_c , old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);

  const sampler_t sampler =
          CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

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
  half4 output0 = read_imageh(bias, sampler, (int2)(out_c, 0));
    half4 output1 = read_imageh(bias, sampler, (int2)(out_c, 0));
    half4 output2 = read_imageh(bias, sampler, (int2)(out_c, 0));
    half4 output3 = read_imageh(bias, sampler, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  half4 output0 = read_imageh(bias, sampler, output_pos0);
    half4 output1 = read_imageh(bias, sampler, output_pos1);
    half4 output2 = read_imageh(bias, sampler, output_pos2);
    half4 output3 = read_imageh(bias, sampler, output_pos3);

#else
  half4 output0 = 0.0f;
  half4 output1 = 0.0f;
  half4 output2 = 0.0f;
  half4 output3 = 0.0f;
#endif

  int max_w_bound = input_c * input_width;
  int burndary_index = input_c * 4 - input_c_origin;
  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x, in_pos_in_one_block0.y);
    half4 input0 = read_imageh(input_image, sampler, pos_in);

    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
    half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
    half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
    half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));

    int bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input0.w = 0.0f;
      } else if (burndary_index==2){
        input0.z = 0.0f;
        input0.w = 0.0f;
      } else if (burndary_index==3){
        input0.y = 0.0f;
        input0.z = 0.0f;
        input0.w = 0.0f;
      }
    }
    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);
    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x, in_pos_in_one_block1.y);
    half4 input1 = read_imageh(input_image, sampler, pos_in);

    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input1.w = 0.0f;
      } else if (burndary_index==2){
        input1.z = 0.0f;
        input1.w = 0.0f;
      } else if (burndary_index==3){
        input1.y = 0.0f;
        input1.z = 0.0f;
        input1.w = 0.0f;
      }
    }
    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    // -------------2--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x, in_pos_in_one_block2.y);
    half4 input2 = read_imageh(input_image, sampler, pos_in);

    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input2.w = 0.0f;
      } else if (burndary_index==2){
        input2.z = 0.0f;
        input2.w = 0.0f;
      } else if (burndary_index==3){
        input2.y = 0.0f;
        input2.z = 0.0f;
        input2.w = 0.0f;
      }
    }
    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    // -------------3--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x, in_pos_in_one_block3.y);
    half4 input3 = read_imageh(input_image, sampler, pos_in);
    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input3.w = 0.0f;
      } else if (burndary_index==2){
        input3.z = 0.0f;
        input3.w = 0.0f;
      } else if (burndary_index==3){
        input3.y = 0.0f;
        input3.z = 0.0f;
        input3.w = 0.0f;
      }
    }

    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);
  }

#ifdef BATCH_NORM
  output0 = output0 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output1 = output1 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output2 = output2 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output3 = output3 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

  if (out_w0 < old_w) {
    write_imageh(output_image, output_pos0, output0);
  }

  if (out_w1 < old_w){
    write_imageh(output_image, output_pos1, output1);
  }

  if (out_w2 < old_w){
    write_imageh(output_image, output_pos2, output2);
  }

  if (out_w3 < old_w){
    write_imageh(output_image, output_pos3, output3);
  }
}

__kernel void conv_1x1_spl_DRELU(
        __private const int global_size_dim0, __private const int global_size_dim1,
        __private const int global_size_dim2, __read_only image2d_t input_image,
        __read_only image2d_t filter,
        __write_only image2d_t output_image, __private const int stride,
        __private const int offset, __private const int input_c,__private const int input_c_origin,
        __private const int dilation,
        __private const int input_width,  /* of one block */
        __private const int input_height, /* of one block */
        __private const int output_width,
        __private const int output_height,
        __private const int old_w
) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

  int outpos_main = mul24(out_c , old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);

  const sampler_t sampler =
          CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

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

  half4 output0 = 0.0f;
  half4 output1 = 0.0f;
  half4 output2 = 0.0f;
  half4 output3 = 0.0f;

  int max_w_bound = input_c * input_width;
  int burndary_index = input_c * 4 - input_c_origin;
  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x, in_pos_in_one_block0.y);
    half4 input0 = read_imageh(input_image, sampler, pos_in);

    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
    half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
    half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
    half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));

    int bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input0.w = 0.0f;
      } else if (burndary_index==2){
        input0.z = 0.0f;
        input0.w = 0.0f;
      } else if (burndary_index==3){
        input0.y = 0.0f;
        input0.z = 0.0f;
        input0.w = 0.0f;
      }
    }
    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);
    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x, in_pos_in_one_block1.y);
    half4 input1 = read_imageh(input_image, sampler, pos_in);

    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input1.w = 0.0f;
      } else if (burndary_index==2){
        input1.z = 0.0f;
        input1.w = 0.0f;
      } else if (burndary_index==3){
        input1.y = 0.0f;
        input1.z = 0.0f;
        input1.w = 0.0f;
      }
    }
    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    // -------------2--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x, in_pos_in_one_block2.y);
    half4 input2 = read_imageh(input_image, sampler, pos_in);

    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input2.w = 0.0f;
      } else if (burndary_index==2){
        input2.z = 0.0f;
        input2.w = 0.0f;
      } else if (burndary_index==3){
        input2.y = 0.0f;
        input2.z = 0.0f;
        input2.w = 0.0f;
      }
    }
    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    // -------------3--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x, in_pos_in_one_block3.y);
    half4 input3 = read_imageh(input_image, sampler, pos_in);
    bound_gap = max_w_bound - pos_in.x - 1;
    if (bound_gap < input_width && bound_gap >= 0){
      if (burndary_index==0){
        // do nothing
      } else if (burndary_index==1){
        input3.w = 0.0f;
      } else if (burndary_index==2){
        input3.z = 0.0f;
        input3.w = 0.0f;
      } else if (burndary_index==3){
        input3.y = 0.0f;
        input3.z = 0.0f;
        input3.w = 0.0f;
      }
    }

    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);
  }


#ifdef RELU
  output0 = activation(output0);
     output1 = activation(output1);
     output2 = activation(output2);
     output3 = activation(output3);
#endif

  if (out_w0 < old_w) {
    write_imageh(output_image, output_pos0, output0);
  }

  if (out_w1 < old_w){
    write_imageh(output_image, output_pos1, output1);
  }

  if (out_w2 < old_w){
    write_imageh(output_image, output_pos2, output2);
  }

  if (out_w3 < old_w){
    write_imageh(output_image, output_pos3, output3);
  }
}



