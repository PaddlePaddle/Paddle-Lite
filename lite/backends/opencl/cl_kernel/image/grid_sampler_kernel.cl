/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

inline void compute_grid_hw(float* grid_x,
                            float* grid_y,
                            CL_DTYPE g1,
                            CL_DTYPE g2,
                            int out_width,
                            int out_height) {
#ifdef ALIGN_CORNER
  *grid_x = (g1 + 1) * (out_width - 1) * 0.5;
  *grid_y = (g2 + 1) * (out_height - 1) * 0.5;
#ifdef BORDER
  *grid_x = fmin(fmax(*grid_x, 0), out_width - 1);
  *grid_y = fmin(fmax(*grid_y, 0), out_height - 1);
#endif
#ifdef REFLECTION
  float x_max = out_width - 1;
  float y_max = out_height - 1;
  // x
  float double_range_x = x_max * 2;
  float grid_x_abs = fabs(*grid_x);
  float extra_x =
      grid_x_abs - (int)(grid_x_abs / double_range_x) * double_range_x;
  *grid_x = fmin(extra_x, double_range_x - extra_x);
  // y
  float double_range_y = y_max * 2;
  float grid_y_abs = fabs(*grid_y);
  float extra_y =
      grid_y_abs - (int)(grid_y_abs / double_range_y) * double_range_y;
  *grid_y = fmin(extra_y, double_range_y - extra_y);
#endif
#else
  *grid_x = (g1 + 1) * out_width * 0.5 - 0.5;
  *grid_y = (g2 + 1) * out_height * 0.5 - 0.5;
#ifdef BORDER
  *grid_x = fmin(fmax(*grid_x, 0), out_width - 1);
  *grid_y = fmin(fmax(*grid_y, 0), out_height - 1);
#endif
#ifdef REFLECTION
  float x_max = out_width - 1;
  float y_max = out_height - 1;
  // x
  float double_range_x = (x_max + 1) * 2;
  float grid_x_abs = fabs(*grid_x + 0.5);
  float extra_x =
      grid_x_abs - (int)(grid_x_abs / double_range_x) * double_range_x;
  *grid_x = fmin(extra_x, double_range_x - extra_x) - 0.5;
  *grid_x = fmin(fmax(*grid_x, 0), x_max);
  // y
  float double_range_y = (y_max + 1) * 2;
  float grid_y_abs = fabs(*grid_y + 0.5);
  float extra_y =
      grid_y_abs - (int)(grid_y_abs / double_range_y) * double_range_y;
  *grid_y = fmin(extra_y, double_range_y - extra_y) - 0.5;
  *grid_y = fmin(fmax(*grid_y, 0), y_max);
#endif
#endif
}

inline void compute_output_with_grid_sampler(__read_only image2d_t input,
                                             __write_only image2d_t output,
                                             float grid_x,
                                             float grid_y,
                                             int out_width,
                                             int out_height,
                                             int out_c,
                                             int out_n,
                                             CL_DTYPE4* out_val,
                                             int2 out_pos) {
#ifdef NEAREST
  int in_ind_w = round(grid_x);
  int in_ind_h = round(grid_y);
  int x_p = out_c * out_width + in_ind_w;
  int y_p = out_n * out_height + in_ind_h;

  if (in_ind_w < 0 || in_ind_w > out_width - 1 || in_ind_h < 0 ||
      in_ind_h > out_height - 1) {
    *out_val = (CL_DTYPE4)(0.0);
  } else {
    *out_val = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x_p, y_p));
  }
#endif
#ifdef BILINEAR
  int xw = floor(grid_x);
  int yn = floor(grid_y);
  int x_p = out_c * out_width + xw;
  int y_p = out_n * out_height + yn;

  float dw = grid_x - xw;
  float de = xw + 1 - grid_x;
  float dn = grid_y - yn;
  float ds = yn + 1 - grid_y;

  CL_DTYPE4 in_nw =
      READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x_p, y_p));
  CL_DTYPE4 in_ne =
      READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x_p + 1, y_p));
  CL_DTYPE4 in_sw =
      READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x_p, y_p + 1));
  CL_DTYPE4 in_se =
      READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x_p + 1, y_p + 1));

  if (xw < 0 || xw > out_width - 1 || yn < 0 || yn > out_height - 1) {
    in_nw = (CL_DTYPE4)(0.0);
  }
  if (xw + 1 < 0 || xw + 1 > out_width - 1 || yn < 0 || yn > out_height - 1) {
    in_ne = (CL_DTYPE4)(0.0);
  }
  if (xw < 0 || xw > out_width - 1 || yn + 1 < 0 || yn + 1 > out_height - 1) {
    in_sw = (CL_DTYPE4)(0.0);
  }
  if (xw + 1 < 0 || xw + 1 > out_width - 1 || yn + 1 < 0 ||
      yn + 1 > out_height - 1) {
    in_se = (CL_DTYPE4)(0.0);
  }
  *out_val = in_nw * (CL_DTYPE4)(de) * (CL_DTYPE4)(ds) +
             in_ne * (CL_DTYPE4)(dw) * (CL_DTYPE4)(ds) +
             in_sw * (CL_DTYPE4)(de) * (CL_DTYPE4)(dn) +
             in_se * (CL_DTYPE4)(dw) * (CL_DTYPE4)(dn);
#endif
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos, *out_val);
}

__kernel void grid_sampler(__read_only image2d_t input,
                           __read_only image2d_t grid,
                           __write_only image2d_t output,
                           __private const int out_height,
                           __private const int out_width,
                           __private const int out_hblks) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_hblks;
  const int out_hblk_id = out_nh % out_hblks;

  int2 coords1, coords2, outpoints;
  coords1.x = out_hblk_id * 2;
  coords1.y = out_n * out_width + out_w;
  coords2.x = coords1.x + 1;
  coords2.y = coords1.y;
  outpoints.x = out_c * out_width + out_w;
  outpoints.y = out_n * out_height + out_hblk_id * 4;

  CL_DTYPE4 g1 = READ_IMG_TYPE(CL_DTYPE_CHAR, grid, SAMPLER, coords1);
  CL_DTYPE4 g2 = READ_IMG_TYPE(CL_DTYPE_CHAR, grid, SAMPLER, coords2);
  float grid_x = 0.f;
  float grid_y = 0.f;
  CL_DTYPE4 out_val;

  compute_grid_hw(&grid_x, &grid_y, g1.x, g2.x, out_width, out_height);
  compute_output_with_grid_sampler(input,
                                   output,
                                   grid_x,
                                   grid_y,
                                   out_width,
                                   out_height,
                                   out_c,
                                   out_n,
                                   &out_val,
                                   outpoints);

  if (out_hblk_id * 4 + 1 < out_height) {  // y
    outpoints.y++;
    compute_grid_hw(&grid_x, &grid_y, g1.y, g2.y, out_width, out_height);
    compute_output_with_grid_sampler(input,
                                     output,
                                     grid_x,
                                     grid_y,
                                     out_width,
                                     out_height,
                                     out_c,
                                     out_n,
                                     &out_val,
                                     (int2)(outpoints.x, outpoints.y));
  }
  if (out_hblk_id * 4 + 2 < out_height) {  // z
    outpoints.y++;
    compute_grid_hw(&grid_x, &grid_y, g1.z, g2.z, out_width, out_height);
    compute_output_with_grid_sampler(input,
                                     output,
                                     grid_x,
                                     grid_y,
                                     out_width,
                                     out_height,
                                     out_c,
                                     out_n,
                                     &out_val,
                                     (int2)(outpoints.x, outpoints.y));
  }
  if (out_hblk_id * 4 + 3 < out_height) {  // w
    outpoints.y++;
    compute_grid_hw(&grid_x, &grid_y, g1.w, g2.w, out_width, out_height);
    compute_output_with_grid_sampler(input,
                                     output,
                                     grid_x,
                                     grid_y,
                                     out_width,
                                     out_height,
                                     out_c,
                                     out_n,
                                     &out_val,
                                     (int2)(outpoints.x, outpoints.y));
  }
}
