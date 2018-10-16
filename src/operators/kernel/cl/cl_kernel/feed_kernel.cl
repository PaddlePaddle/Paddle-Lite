#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void feed(__global float *in, __write_only image2d_t outputImage,int h,int w)
 {
        int i = get_global_id(0);
        int j = get_global_id(1);
        half4 pixel;
        pixel.x = convert_half(in[(i * w + j)]);
        pixel.y = convert_half(in[h * w + (i * w + j)]);
        pixel.z = convert_half(in[2 * h * w + (i * w + j)]);
        pixel.w = 0.0;
        int2 coords;
        coords.x = j;
        coords.y = i;

        write_imageh(outputImage,coords,pixel);
 }
