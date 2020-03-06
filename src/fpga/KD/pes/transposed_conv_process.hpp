#pragma once

#ifndef transposed_conv_process_hpp
#define transposed_conv_process_hpp

#include <string.h>
#include <cmath>
#include <vector>

#include "../float16.hpp"
#include "../tensor.hpp"
#include "conv_process.hpp"

namespace paddle_mobile {
namespace zynqmp {
//  /*
//     calculate sub padding number
//     */
//     int calc_sub_pad(int filter_axis, int pad, int stride) {
//             if (stride == 0 || ((filter_axis - pad - 1) < 0)) {
//                     // PADDLE_MOBILE_ENFORCE(false, "Wrong deconv parameters");
//             }
//             return (filter_axis - pad - 1) / stride;
//     }

//     int get_sub_filter_axis(int filter_axis, int stride) {
//             return (filter_axis / stride);
//     }

//     int get_sub_out_axis(int image_axis, int sub_pad, int sub_filter_axis) {
//             return ((image_axis + 2 * sub_pad - sub_filter_axis) + 1);
//     }

//     /*
//     (filter_width-pad,filter_width-pad) is the first pixel of sub-pixel image
//     position. so the omit rows or columns is (stride - )
//     */
//     int deconv_get_omit(int stride, int filter_width, int pad) {
//         // PADDLE_MOBILE_ENFORCE(filter_width > pad, "Wrong deconv parameters");
//         int idx;
//         bool flag = false;
//         for (idx = 1; idx <= stride; ++idx) {
//                 int j = idx;
//                 for (; j <= filter_width;) {
//                         if (j == filter_width - pad) {
//                                 flag = true;
//                                 break;
//                         }
//                         j = j + stride;
//                 }
//                 if (flag) {
//                         break;
//                 }
//         }
//         return (stride - idx);
//     }

    /**
    filter data in PaddlePaddle is CNHW format.
    this function convert it into NCHW format.
    */
    void inline convert_cnhw_to_nchw(Tensor* cnhw, Tensor* nchw) {
        // cnhw->saveToFile("cnhw", true);
        // cnhw->shape().setLayoutType(CNHW);
        Shape& cnhw_shape = cnhw->shape();
        Shape shape(NCHW, {cnhw_shape.channel(), cnhw_shape.num(),
                cnhw_shape.height(), cnhw_shape.width()});
        float* nchw_data = nchw->mutableData<float>(FP32,  shape);
        float* cnhw_data = cnhw->data<float>();

        int hw = shape.height() * shape.width();
        int nhw = shape.num() * hw;
        int chw = shape.channel() * hw;

        // float* filter_data = param_.filter->data<float>();
        int index = 0;
        for (int c = 0; c < shape.channel(); c++) {
            for (int n = 0; n < shape.num(); n++) {
                // for (int h = 0; h < shape.height(); h++) {
                //     for (int w = 0; w < shape.width(); w++) {
                //         int dst_index = 
                //         index++;
                //     }
                // }

               
                float* dst = nchw_data + c * hw + n * chw;
                // float* src = cnhw_data + n * chw + c * hw;
                float* src = cnhw_data + index;
                memcpy(dst, src, hw * sizeof(float));

                index += hw;
            }
        }
        // nchw->saveToFile("nchw", true);

        // exit(-1);
    }

    template<typename T>
    void inline chw_to_hwc2(Tensor* chw, Tensor* hwc) {
        Shape& shape = chw->shape();

        T* x = chw->data<T>();
        T* y = hwc->data<T>();

        int channel = shape.channel();
        int height = shape.height();
        int width = shape.width();
        int index = 0;
        int wc = width * channel;
        for (int c = 0; c < channel; c++) {
            for (int h = 0; h < height; h++) {
                int offset_height = h * wc;
                for (int w = 0; w < width; w++) {
                    int dst_index = offset_height + w * channel + c;
                    y[dst_index] = x[index];
                    // std::cout << "dst_index:" << dst_index << " index:" << index << std::endl;
                    index++;
                }
            }
        }
    }

    template<typename T>
    void inline hwc_to_chw(Tensor* hwc, Tensor* chw) {
        Shape& shape = chw->shape();

        T* x = hwc->data<T>();
        T* y = chw->data<T>();

        int channel = shape.channel();
        int height = shape.height();
        int width = shape.width();
        int index = 0;
        int wc = width * channel;
        int hw = height * width;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channel; c++) {
                // int offset_height = h * wc;
                    int dst_index = c * hw + h * width + w;
                    y[dst_index] = x[index];
                    // std::cout << "dst_index:" << dst_index << " index:" << index << std::endl;
                    index++;
                }
            }
        }
    }

    /**
    inverse data in one HW plane,take 3x3 filter for example
    0<->8, 1<-->7, 2<-->6, 3<-->5
    */
    void inline inverse_filter(Tensor* tensor) {
        float* data = tensor->data<float>();
        Shape& shape = tensor->shape();
        int hw = shape.height() * shape.width();
        int chw = shape.channel() * hw;

        int hw_last_index = hw - 1;
        for (int n = 0; n < shape.num(); n++) {
            for (int c = 0; c < shape.channel(); c++) {
                float* hw_start = data + n * chw + c * hw;
                float tmp_data;
                for (int i = 0; i < hw / 2; i++) {
                    tmp_data = hw_start[i];
                    hw_start[i] = hw_start[hw_last_index - i];
                    hw_start[hw_last_index - i] = tmp_data;
                }
            }
        }
    }

//  void fill_sub_filters(ConvParam* param, Tensor* filter) {
//         Tensor* input = param->input;
//         int sub_conv_number = param->strides[0];

//         int kernel_num = filter->shape().num();
//         int height = filter->shape().height();
//         int width = filter->shape().width();

//         int sub_num = kernel_num * sub_conv_number;
//         int sub_h = height / sub_conv_number;
//         int sub_w = width / sub_conv_number;

//         float* filter_data = filter->data<float>();

//         std::vector<float> filter_scales;

//         int channel = filter->shape().channel(); //TODO
//         for (int i = 0; i < sub_conv_number; i++) {
//             int sub_num = kernel_num * sub_conv_number;
//             int sub_h = height / sub_conv_number;
//             int sub_w = width / sub_conv_number;
//             Shape shape(NHWC, {sub_num, sub_h, sub_w, channel});
//             BasicConvParam* basic_conv_param = new BasicConvParam();


//             basic_conv_param->filter.mutableData<int8_t>(INT8, shape);
//             Tensor float_tensor;
//             float* sub_filter_data = float_tensor.mutableData<float>(FP32, shape);

//             for (int nn = 0; nn < sub_num; ++nn) { //TODO optimize theese code;
//                 int ni = nn % kernel_num;
//                 int woff = sub_conv_number - 1 - (nn / kernel_num);  //
//                 for (int hh = 0; hh < sub_h; ++hh) {
//                     int hi = hh * sub_conv_number + i % sub_conv_number;
//                     for (int ww = 0; ww < sub_w; ++ww) {
//                         int wi = ww * sub_conv_number + woff;  // 1 0
//                         int sidx = ((nn * sub_h + hh) * sub_w + ww) * channel;   //
//                         int kidx = ((ni * height + hi) * width + wi) * channel;  //
//                         fpga_copy(
//                                 sub_filter_data + i * sub_h * sub_w * channel * sub_num + sidx,
//                                 filter_data + kidx, channel * sizeof(float));
//                     }
//                 }
//             }

//             format_filter(&float_tensor, &(basic_conv_param->filter), 1, filter_scales,1);

//             ConvArgs& args = basic_conv_param->args;


//             int sub_pad = calc_sub_pad(filter->shape().width(), param.padding[1], param.strides[1]);
// //  deconv_filter::deconv_calc_sub_pad((int)filter->dims()[3],  // NOLINT
// //                                     padding_w, stride_w);


//             auto sub_filter_width = (uint32_t)deconv_filter::deconv_get_sub_filter_axis(
//                     (int)filter->dims()[3], stride_w);  // NOLINT

//             auto sub_output_width = (uint32_t)get_sub_out_axis(
//                     (int)input->shape().width(), sub_pad, sub_filter_width);  // NOLINT
//             auto sub_output_height = (uint32_t)deconv_filter::deconv_get_sub_out_axis(
//                     (int)input->shape().height(), sub_pad, sub_filter_width);  // NOLINT


//             auto omit_size = (uint32_t)deconv_filter::deconv_get_omit(
//                     stride_w, (int)filter->shape().width(), padding_w);  // NOLINT

//             auto sub_channels = (int)input->shape().channel();  // NOLINT


//             args.group_num = param.groups;
//             // TODO relu by inplace
//             // args.relu_enabled = param.relu.enabled;
//             args.sb_address = conv_param->scaleBias.data<float16>();
//             // args.sb_address = bs_data;
//             args.kernel.stride_h = 1;
//             args.kernel.stride_w = 1;
//             args.kernel.height = sub_h;
//             args.kernel.width = sub_w;

//             args.filter_address = conv_param->filter.data<int8_t>();
//             args.filter_num = sub_num;
//             args.filter_scale_address = conv_param->filter.scale();
//             args.image.address = input->data<void>();
//             args.image.scale_address = input->scale();
//             args.image.channels = input->shape().channel();
//             args.image.width = input->shape().width();
//             args.image.height = input->shape().height();
//             args.image.pad_width = sub_pad;
//             args.image.pad_height = sub_pad;
//              // TODO dilations[0] = dilations[1]
//             args.dilation = param.dilations[0];

//             args.output.address = out_address;
//             args.output.scale_address = out_scale_address;


//                 param->splitParams().push_back(basic_conv_param);
//         }
//  }

//  void inline format_transposed_filter(Tensor* tensor) {}

//     inline int fill_transposed_split_arg(const ConvParam& c_param) {
//         ConvParam& param = const_cast<ConvParam&>(c_param);
//         Tensor nchw;
//         convert_cnhw_to_nchw(param.filter, &nchw);
//         inverse_filter(&nchw);
//         fill_sub_filters(&param, &nchw);
//     }

}  // namespace zynqmp
}  // namespace paddle_mobile

#endif /* transposed_conv_process_hpp */
