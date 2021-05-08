/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <vector>

#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/concat_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/scale_pe.hpp"
#include "lite/backends/fpga/KD/pes/split_pe.hpp"
#include "lite/backends/fpga/KD/pes/transposed_conv_process.hpp"

namespace paddle {
namespace zynqmp {

class TransposedConvPE : public PE {
 public:
  bool init() {
    Tensor* filter = param_.filter;

    // Paddle Lite treats each params layout as [N, C, H, W]
    // As for transposed conv, the actual layout in memory is [C, N, H, W]
    // So only layout type should be changed here.
    filter->shape().setLayoutType(LayoutType::CNHW);

    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);

    return true;
  }

  void apply() {
    int kernel_width = param_.filter->shape().width();
    int kernel_height = param_.filter->shape().height();
    int stride_width = param_.strides[0];
    int padding_width = param_.paddings[0];

    if (kernel_width % stride_width == 0) {
      sub_filter_ena_ = true;
    } else {
      sub_filter_ena_ = false;
    }
    // pad inputs on ZU3 devices
    if (DLEngine::get_instance().isZU3()) {
      sub_filter_ena_ = false;
    }
    sub_filter_ena_ = false;

    ConvParam& conv_param = pe_.param();
    // just for test
    static int counter = 0;
    std::string ss_filter;
    ss_filter = std::to_string(counter).append("transposed_filter");
    std::cout << "current feed filter is " << ss_filter << std::endl;
    std::string ss_bias;
    ss_bias = std::to_string(counter).append("transposed_bias");
    std::cout << "current feed bias is " << ss_bias << std::endl;
    std::string ss_scale;
    ss_scale = std::to_string(counter).append("transposed_scale");
    std::cout << "current feed scale is " << ss_scale << std::endl;

    param_.filter->readFloatFromFile(ss_filter);
    param_.bias()->readFloatFromFile(ss_bias);
    param_.scale()->readFloatFromFile(ss_scale);
    std::cout << "read filter, scale and bias" << std::endl;      

    ++counter;
    // end test
    convert_cnhw_to_nchw(param_.filter, &filter_);
    inverse_filter(&filter_);

    if (sub_filter_ena_) {
      omit_size_ = deconv_get_omit(stride_width, kernel_width, padding_width);
      sub_conv_number_ = param_.strides[0];
      fill_sub_filters(&param_, &filter_);
      conv_param.input = param_.input;
      conv_param.output = param_.output;
      conv_param.filter = param_.filter;
      for(auto basic : const_cast<ConvParam&>(param_).splitParams()) {
        basic->args.inplace.active_param.type = param_.activeParam.type;   
      	conv_param.splitParams().push_back(basic);
      }
      conv_param.deconv = true;
      conv_param.activeParam.type = param_.activeParam.type;
      conv_param.cpu_concat = param_.cpu_concat;
    } else {
      Shape& input_shape = param_.input->shape();
      int padded_height = input_shape.height() +
                          (input_shape.height() - 1) * (param_.strides[0] - 1);
      int padded_width = input_shape.width() +
                         (input_shape.width() - 1) * (param_.strides[1] - 1);
      Shape padded_shape(NCHW,
                         {input_shape.num(),
                          input_shape.channel(),
                          padded_height,
                          padded_width});

      int ph = param_.filter->shape().height() - param_.paddings[0] - 1;
      int pw = param_.filter->shape().width() - param_.paddings[1] - 1;

      padded_input_.mutableData<float16>(FP16, padded_shape);
      conv_param.input = &padded_input_;
      conv_param.output = param_.output;
      conv_param.filter = &filter_;
      conv_param.strides = {1, 1};
      conv_param.paddings = {ph, pw};
      conv_param.kernelSize = {kernel_height, kernel_width};
      conv_param.dilations = {1, 1};
      conv_param.deconv = false;
      conv_param.activeParam.type = param_.activeParam.type;
      conv_param.scale()->mutableData<float>(FP32, param_.scale()->shape());
      conv_param.scale()->copyFrom(param_.scale());
      conv_param.bias()->mutableData<float>(FP32, param_.bias()->shape());
      conv_param.bias()->copyFrom(param_.bias());
    }
    pe_.init();
    pe_.apply();
  }

  template <typename T>
  void pad_input() {
    param_.input->syncToCPU();
    T* input_data = param_.input->data<T>();
    int channel = param_.input->shape().channel();
    int in_wc = param_.input->shape().width() * channel;
    int o_wc = padded_input_.shape().width() * channel;

    T* data = padded_input_.data<T>();
    int oh = param_.input->shape().height();
    int ow = param_.input->shape().width();
    memset(data, 0, padded_input_.memorySize());

    for (int h = 0; h < oh; h++) {
      for (int w = 0; w < ow; w++) {
        T* src = input_data + h * in_wc + w * channel;
        T* dst = data + (h)*param_.strides[0] * o_wc +
                 (w) * (param_.strides[1]) * channel;
        memcpy(dst, src, channel * sizeof(T));
      }
    }

    padded_input_.flush();
    padded_input_.copyScaleFrom(param_.input);
    padded_input_.copyMaxFrom(param_.input);
  }

  bool dispatch() {
    if (sub_filter_ena_ == false) {
      pad_input<float16>();
    }
    // just for test
    static int counter = 0;
    if(counter == 0) {
      param_.input->readHalfFromFile("transposed_input");
      param_.input->saveToFile("checkinput", true);
      std::cout << "read input" << std::endl;
    }
    ++counter;
    // end test
    bool ret = pe_.dispatch();
    if(ret == true) {
      if(sub_filter_ena_ && !param_.cpu_concat) {
        float max_val = 0.0;
        for (auto conv_param : param_.splitParams()) {
            std::cout << half_to_float(conv_param->output_max) << std::endl;
          max_val = std::max(max_val, half_to_float(conv_param->output_max));
        }
        param_.output->max()[0] = float_to_half(max_val);
      }
      else if(sub_filter_ena_) {
        // all the split sub filters are concated by cpu
        std::cout << "cpu concat transposed";
        param_.output->setOffset(0);
        splited_sub_res_concat();
      }
      int oc = param_.output->shape().channel();
      int ow = param_.output->shape().width();
      int off_addr = omit_size_ * oc * ow;
      param_.output->unalignImage();
      param_.output->setOffset(off_addr);

    }

    // param_.output->saveToFile("final2", true);
    return ret;
  }

  void splited_sub_res_concat() {
    // final output config
    Tensor* final_output = param_.output;
    float16* final_dst = final_output->mutableData<float16>();

    float16* dst_cur = final_dst;

    float16 final_max = float_to_half(0.0);
    int ow = final_output->shape().width();
    int oc = final_output->shape().channel();
    int dst_stride_in_wc = align_to_x(oc * ow, 16);
    // std::cout << "final output w is " << ow << "final ouput c is " << oc << std::endl;
    // split sub filter config
    auto conv_params = param_.splitParams();
    int total_res_num = conv_params.size();
    int res_num_per_sub = total_res_num / sub_conv_number_;
    int omit_low = oc * omit_size_;
    int omit_high = oc * (sub_conv_number_ - omit_size_);
    int accum_c = 0;

    // for every sub filter outputs
    for(int sub_idx = 0; sub_idx < sub_conv_number_; ++sub_idx) {
      // std::cout << "current sub idx is " << sub_idx << std::endl;
      // for every split sub outputs, clear accum_c
      accum_c = 0;
      // std::cout << "current accum c is " << accum_c << std::endl;

      for(int idx_in_sub = 0; idx_in_sub < res_num_per_sub; ++idx_in_sub) {
        // std::cout << "current idx in sub is " << idx_in_sub << std::endl;
        // dst start position for split sub output
        float16* dst_start_hwc = final_dst + sub_idx * dst_stride_in_wc;

        // src data and config for each split sub output
        auto each_conv_param = conv_params[sub_idx * res_num_per_sub + idx_in_sub];
        float16 cur_max = each_conv_param->output_max;
        final_max = std::max(cur_max, final_max);

        each_conv_param->output.invalidate();

        float16* src_start = each_conv_param->output.data<float16>();
        int each_C = each_conv_param->output.shape().channel();
        int each_W = each_conv_param->output.shape().width();
        int each_H = each_conv_param->output.shape().height();

        int src_stride_in_wc = align_to_x(each_C * each_W, 16);


        // for every wc in each split sub output
        for(int h = 0; h < each_H; ++h) {

          float16* src_hwc = src_start + h * src_stride_in_wc;
          float16* dst_cur_hwc = dst_start_hwc + sub_conv_number_ * h * dst_stride_in_wc;


          // when w = 0
          // check the remain fill length of the first oc
          int dst_w_start_without_omit = accum_c / oc;
          if(dst_w_start_without_omit < omit_size_) {
            // std::cout << "dst_w_start_without_omit is " << dst_w_start_without_omit << std::endl;
            int dst_w_end_without_omit = accum_c + each_C;
            if(dst_w_end_without_omit > omit_size_) {
              // fill part of oc
              // std::cout << "need fill part" << std::endl;
              int part_fill_start = omit_size_ * oc - accum_c;
              memcpy(dst_cur_hwc, src_hwc + part_fill_start, (each_C - part_fill_start) * sizeof(float16));
            }
          } else {
            // fill a complete oc, idx[hwc], idx[wc], idx[c]
            // std::cout << "fill complete" << std::endl;
            float16* dst_fill_copy_start = dst_cur_hwc + (dst_w_start_without_omit - omit_size_) * oc + accum_c % oc;
            memcpy(dst_fill_copy_start, src_hwc, each_C * sizeof(float16));
          }

          // Because the first oc might be omitted, in order to be simple, just omit the first
          float16* dst_start_wc = dst_cur_hwc + (sub_conv_number_ - omit_size_ + idx_in_sub) * oc;
          dst_cur = dst_start_wc + accum_c % oc;
          for(int w = 1; w < each_W - 1; ++w) {
            float16* src_wc = src_hwc + w * each_C;
            memcpy(dst_cur, src_wc, each_C * sizeof(float16));
            dst_cur += oc * sub_conv_number_;
          }

          // when w = each_W - 1
          int start_c = accum_c;
          int end_c = accum_c + each_C;
          float16* src_wc = src_hwc + (each_W - 1) * each_C;
          start_c = std::min(start_c, omit_high) - accum_c;
          end_c = std::min(end_c, omit_high) - accum_c;
          if(start_c < end_c) {
            memcpy(dst_cur, src_wc, (end_c - start_c) * sizeof(float16));
          }


        }// end each_H

        // accumulate channel

        accum_c += each_C;


      }// end each split



    }// end each sub

    final_output->max()[0] = final_max;
    final_output->flush();
    // final_output->saveToFile("final", true);

  }

  ConvParam& param() { return param_; }

 private:
  ConvParam param_;
  ConvPE pe_;
  bool sub_filter_ena_;
  int omit_size_;
  int sub_conv_number_;
  Tensor padded_input_;
  Tensor filter_;
};

}  // namespace zynqmp
}  // namespace paddle
