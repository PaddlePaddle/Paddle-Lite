#pragma once



namespace paddle_mobile {
namespace fpga {
namespace deconv_filter {


void deconv_inverse_filter(float** data_in, int num, int channel, int width, int height);
int deconv_calc_sub_pad(int filter_axis, int pad, int stride);
int deconv_get_sub_filter_num(int filter_num, int stride);
int deconv_get_sub_filter_axis(int filter_axis, int stride);
int deconv_get_sub_out_axis(int image_axis, int sub_pad, int sub_filter_axis);
int deconv_get_omit(int stride, int filter_width, int pad);
void deconv_get_sub_filter(char** data_in, int height, int width, int sub_conv_n, int kernel_num, int channel );
void deconv_format_filter(float** data_in, int num, int channel, int height,
                      int width, int group_num, float max,int stride);
void deconv_NC_convert(float**filter_in, int kernel_num, int channels, int hw);

}  // namespace deconv_filter
}  // namespace fpga
}  // namespace paddle_mobile