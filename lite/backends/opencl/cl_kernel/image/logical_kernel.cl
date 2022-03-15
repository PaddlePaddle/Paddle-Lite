
#include <cl_common.h>

__kernel void xor(__read_only image2d_t input_x,
                      __read_only image2d_t input_y,
                      __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height
  CL_DTYPE4 in_x = READ_IMG_TYPE(CL_DTYPE_CHAR, input_x, SAMPLER, (int2)(x, y));
  CL_DTYPE4 in_y = READ_IMG_TYPE(CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(x, y));
  CL_DTYPE4 in;
  CL_DTYPE4 in_=0;
    int4 a=convert_int4(abs(isnotequal(in_x,in_)));
     int4 b=convert_int4(abs(isnotequal(in_y,in_)));
     a=(a!=b)?1:0;
  in=convert_half4(a);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}


