


__kernel void conv_3x3(__global float* in, __global float* out) {
     int num = get_global_id(0);
     out[num] = in[num] * 0.1  + 102;
 }
