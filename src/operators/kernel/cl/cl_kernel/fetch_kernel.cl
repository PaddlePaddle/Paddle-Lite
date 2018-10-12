#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void fetch(__private const int in_height,
                    __private const int in_width,
                    __private const int size_ch,
                    __private const int size_block,
                    __private const int size_batch,
                    __read_only image2d_t input,
                    __global float* out) {
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
  out[index + size_ch] = convert_float(in.y);
  out[index + size_ch * 2] = convert_float(in.z);
  out[index + size_ch * 3] = convert_float(in.w);
}
