
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define MIN_VALUE -FLT_MAX
__kernel void density_prior_box(__write_only image2d_t output_boxes,
                                __write_only image2d_t output_variances,
                                __global float *densities,
                                __private const float step_h,
                                __private const float step_w,
                                __private float variances0,
                                __private float variances1,
                                __private float variances2,
                                __private float variances3,
                                __private float offset,
                                __private int den_and_fix_size,
                                __private int img_width,
                                __private int img_height,
                                __private int C,
                                __private int num_density,
                                __private int step_average,
                                __private int input_width,
                                __private int wid,
                                __private int fix_ratio_size
                                ){

                                const int out_c = get_global_id(0);
                                const int out_w = get_global_id(1);
                                const int out_nh = get_global_id(2);
                                int2 output_pos;
                                output_pos.x = out_c * 4 + out_w;
                                output_pos.y = out_nh;
                                half4 output;
                                half4 variances;
                                for (int c = 0; c < 4; c++) {
                                    int idx = out_nh % num_density;
                                    int input_h = out_nh / num_density;
                                    int input_w = out_c * 4 + c;
                                    int density_idx;
                                    int density;
                                    int ratio_idx;
                                    int density_i;
                                    int density_j;
                                    int sum = 0;
                                    int pre_sum = 0;
                                    for (int i = 0; i < den_and_fix_size; i++) {
                                        pre_sum = sum;
                                        density = densities[i];
                                        sum += density * density * fix_ratio_size;
                                        if (idx < sum) {
                                            density_idx = i;
                                            break;
                                        }
                                    }
                                    idx = idx - pre_sum;
                                    ratio_idx = idx / (density * density);
                                    idx = idx % (density * density);
                                    density_i = idx / density;
                                    density_j = idx % density;
                                    half fixed_size = densities[den_and_fix_size + density_idx];
                                    half ratio = densities[2 * den_and_fix_size + ratio_idx];
                                    half box_width = fixed_size * ratio;
                                    half box_height = fixed_size / ratio;
                                    int shift = step_average / density;
                                    half center_x;
                                    half center_y;
                                    center_x = (input_w + offset) * step_w;
                                    center_x = center_x - step_average / 2.0 + shift / 2.0;
                                    center_x = center_x + density_j * shift;
                                    center_y = (input_h + offset) * step_h;
                                    center_y = center_y - step_average / 2.0 + shift / 2.0;
                                    center_y = center_y + density_i * shift;
                                    half4 box;
                                    box.x = (center_x - box_width / 2.0) / img_width;
                                    box.y = (center_y - box_height / 2.0) / img_height;
                                    box.z = (center_x + box_width / 2.0) / img_width;
                                    box.w = (center_y + box_height / 2.0) / img_height;
                                    box.x = max((float)box.x, 0.0);
                                    box.y = max((float)box.y, 0.0);
                                    box.z = min((float)box.z, 1.0);
                                    box.w = min((float)box.w, 1.0);
                                    half res;
                                    half var;
                                    if (out_w == 0) {
                                        res = box.x;
                                        var = convert_half(variances0);
                                    } else if (out_w == 1) {
                                        res = box.y;
                                        var = convert_half(variances1);
                                    } else if (out_w == 2) {
                                        res = box.z;
                                        var = convert_half(variances2);
                                    } else if (out_w == 3) {
                                        res = box.w;
                                        var = convert_half(variances3);
                                    }
                                    variances.x = var;
                                    variances.y = var;
                                    variances.z = var;
                                    variances.w = var;
                                    if (c == 0) {
                                        output.x = res;
                                    } else if (c == 1) {
                                        output.y = res;
                                    } else if (c == 2) {
                                        output.z = res;
                                    } else if (c == 3) {
                                        output.w = res;
                                    }
                                }

                                write_imageh(output_boxes, (int2)(output_pos.x, output_pos.y), output);

                                write_imageh(output_variances, (int2)(output_pos.x, output_pos.y), variances);

}