
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void relu(__read_only image2d_t input,
                   __write_only image2d_t output){

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  half4 in = read_imageh(input, sampler, (int2)(x, y));
  in = max((half4)(0.0), in);
  write_imageh(output, (int2)(x, y), in);
}