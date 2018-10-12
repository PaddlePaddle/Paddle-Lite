#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void batchnorm(__private const int out_height,
                        __private const int out_width,
                        __read_only image2d_t input,
                        __read_only image2d_t new_scale,
                        __read_only image2d_t new_bias,
                        __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  half4 new_scale = read_imageh(bn_scale, sampler, (int2)(out_c, 0));
  half4 new_bias = read_imageh(bn_bias, sampler, (int2)(out_c, 0));

  int pos_x = mad24(out_c, out_width, out_w);
  half4 in = read_imageh(input, sampler, (int2)(pos_x, out_nh));
  half4 out = mad(in, new_scale, new_bias);

  write_imageh(output, (int2)(pos_x, nh), out);
}
