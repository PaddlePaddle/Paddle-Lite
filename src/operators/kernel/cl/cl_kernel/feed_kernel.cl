__kernel void feed(__global float* in, __write_only image2d_t outputImage,int h,int w)
 {
     int j = get_global_id(0);
     int i = get_global_id(1);
     float4 pixel;
     pixel.x = in[(i * w + j)];
     pixel.y = in[h * w + (i * w + j)];
     pixel.z = in[2 * h * w + (i * w + j)];
     pixel.w = 0;
     int2 coords;
     coords.x = j;
     coords.y = i;

     write_imagef(outputImage,coords,pixel);
 }
