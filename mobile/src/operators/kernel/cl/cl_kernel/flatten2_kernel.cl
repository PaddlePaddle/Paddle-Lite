

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


__kernel void flatten2(__read_only image2d_t input_img,
                       __write_only image2d_t output_img,
                       __private int out_width,
                       __private int in_width,
                       __private int in_height,
                       __private int in_C
                      ){

                        const int out_c = get_global_id(0);
                        const int out_w = get_global_id(1);
                        const int out_nh = get_global_id(2);

                        const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                    CLK_ADDRESS_CLAMP |
                                                    CLK_FILTER_NEAREST;

                        int2 output_pos;
                        output_pos.x = out_c * out_width + out_w;
                        output_pos.y = out_nh;

                        int channel_size = in_width * in_height;

                        int in_c = output_pos.x / channel_size / 4;
                        int2 input_pos;
                        input_pos.x = (output_pos.x % in_width) + (in_c * in_width);
                        input_pos.y = (output_pos.x % channel_size) / in_width + out_nh * in_height;
                        half4 input_data = read_imageh(input_img, sampler, input_pos);

                        half4 output_data;
                        int in_c_offset = output_pos.x / channel_size % 4;
                        if(in_c_offset == 0){
                            output_data.x = input_data.x;
                        } else if(in_c_offset == 1){
                            output_data.x = input_data.y;
                        } else if(in_c_offset == 2){
                            output_data.x = input_data.z;
                        } else if(in_c_offset == 3){
                            output_data.x = input_data.w;
                        }

                        write_imageh(output_img, output_pos, output_data);
}

