#include <cl_common.h>

// axis=0
__kernel void gather_axis0(__read_only image2d_t input,
                           __read_only image2d_t index,
                           __write_only image2d_t outputImage) {
  int y = get_global_id(0);
  int x = get_global_id(1);
  int2 coords_l;
  coords_l.x = y;  // w
  coords_l.y = 0;  // h
  CL_DTYPE4 id = READ_IMG_TYPE(CL_DTYPE_CHAR, index, SAMPLER, coords_l);
  int2 coords;
  coords.y = id.x;

  int2 coords_;
  coords_.y = y;

  coords.x = x;
  coords_.x = x;
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords_, in);
}
__kernel void gather_axis1(__read_only image2d_t input,
                           __read_only image2d_t index,
                           __write_only image2d_t outputImage) {
  int x = get_global_id(0);  // index
  int y = get_global_id(1);  // h
  int2 coords_l;
  coords_l.x = x;  // w
  coords_l.y = 0;  // h
  CL_DTYPE4 id = READ_IMG_TYPE(CL_DTYPE_CHAR, index, SAMPLER, coords_l);
  int2 coords;
  coords.x = id.x;
  coords.y = y;
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
  int2 coords_;
  coords_.x = x;
  coords_.y = y;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords_, in);
}
